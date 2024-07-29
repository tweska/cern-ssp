#include "CUDAHelpers.cuh"

#include "RHnCUDA.cuh"

namespace ROOT {
namespace Experimental {

////////////////////////////////////////////////////////////////////////////////
/// CUDA kernels
////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////
/// Device kernels for incrementing a bin.

/// @brief Get the bin for a specific axis
/// @param x The input coordinate
/// @param binEdges Array with edges of the histogram bins per axis
/// @param binEdgesIdx Index to the start of the edges array for the current axis in binEdges
/// @param nBins Number of bins for the current axis
/// @param xMin Minimum value of the histogram edges of the current axis
/// @param xMax Maximum value of the histogram edges of the current axis
/// @return The bin for coordinate x
__device__ inline int FindFixBin(double x, const double *binEdges, int binEdgesIdx, int nBins, double xMin, double xMax)
{
   int bin;

   if (x < xMin) { // underflow
      bin = 0;
   } else if (!(x < xMax)) { // overflow  (note the way to catch NaN)
      bin = nBins + 1;
   } else {
      if (binEdgesIdx < 0) { // fixed bins
         bin = 1 + int(nBins * (x - xMin) / (xMax - xMin));
      } else { // variable bin sizes
         bin = 1 + CUDAHelpers::BinarySearch(nBins + 1, &binEdges[binEdgesIdx], x);
      }
   }

   return bin;
}

// Use Horner's method to calculate the bin in an n-Dimensional array.
/// @brief Get the bin for each coordinate. Uses Horner's method to calculate the combined bin for the flat histogram
///        array in case of n-Dimensional arrays.
/// @param tid Current CUDA thread ID
/// @param binEdges Array with bin edges per axis, can be length zero if the bins are fixed size instead of variable
/// @param binEdgesIdx Index to the start of array for each axis in binEdges
/// @param nBinsAxis Number of bins per axis including under/overflow bins
/// @param xMin Minimum edge value per axis
/// @param xMax Maximum edge value per axis
/// @param coords Input coordinate values, in the form of xxx,yyy,zzz in case of multidimensional histograms
/// @param bulkSize Number of coordinates
/// @param mask Mask on input coordinates that are out of bounds
template <unsigned int Dim>
__device__ inline int GetBin(size_t tid, double *binEdges, int *binEdgesIdx, int *nBinsAxis, double *xMin, double *xMax,
                             double *coords, size_t bulkSize, bool *mask)
{
   auto bin = 0;
   for (int d = Dim - 1; d >= 0; d--) {
      auto *x = &coords[d * bulkSize];
      auto binD = FindFixBin(x[tid], binEdges, binEdgesIdx[d], nBinsAxis[d] - 2, xMin[d], xMax[d]);
      mask[tid] *= binD > 0 && binD < nBinsAxis[d] - 1;

      // if (binD < 0) {
      //    return -1;
      // }

      bin = bin * nBinsAxis[d] + binD;
   }

   return bin;
}

///////////////////////////////////////////
/// Device kernels for incrementing a bin.

template <typename T>
__device__ inline void AddBinContent(T *histogram, int bin, double weight)
{
   atomicAdd(&histogram[bin], (T)weight);
}

// TODO:
// template <>
// __device__ inline void AddBinContent(char *histogram, int bin, char weight)
// {
//    int newVal = histogram[bin] + int(weight);
//    if (newVal > -128 && newVal < 128) {
//       atomicExch(&histogram[bin], (char) newVal);
//       return;
//    }
//    if (newVal < -127)
//       atomicExch(&histogram[bin], (char) -127);
//    if (newVal > 127)
//       atomicExch(&histogram[bin], (char) 127);
// }

template <>
__device__ inline void AddBinContent(short *histogram, int bin, double weight)
{
   // There is no CUDA atomicCAS for short so we need to operate on integers... (Assumes little endian)
   short *addr = &histogram[bin];
   int *addrInt = (int *)((char *)addr - ((size_t)addr & 2));
   int old = *addrInt, assumed, newVal, overwrite;

   do {
      assumed = old;

      if ((size_t)addr & 2) {
         newVal = (assumed >> 16) + (int)weight; // extract short from upper 16 bits
         overwrite = assumed & 0x0000ffff;       // clear upper 16 bits
         if (newVal > -32768 && newVal < 32768)
            overwrite |= (newVal << 16); // Set upper 16 bits to newVal
         else if (newVal < -32767)
            overwrite |= 0x80010000; // Set upper 16 bits to min short (-32767)
         else
            overwrite |= 0x7fff0000; // Set upper 16 bits to max short (32767)
      } else {
         newVal = (((assumed & 0xffff) << 16) >> 16) + (int)weight; // extract short from lower 16 bits + sign extend
         overwrite = assumed & 0xffff0000;                          // clear lower 16 bits
         if (newVal > -32768 && newVal < 32768)
            overwrite |= (newVal & 0xffff); // Set lower 16 bits to newVal
         else if (newVal < -32767)
            overwrite |= 0x00008001; // Set lower 16 bits to min short (-32767)
         else
            overwrite |= 0x00007fff; // Set lower 16 bits to max short (32767)
      }

      old = atomicCAS(addrInt, assumed, overwrite);
   } while (assumed != old);
}

template <>
__device__ inline void AddBinContent(int *histogram, int bin, double weight)
{
   int old = histogram[bin], assumed;
   long newVal;

   do {
      assumed = old;
      newVal = max(long(-INT_MAX), min(assumed + long(weight), long(INT_MAX)));
      old = atomicCAS(&histogram[bin], assumed, newVal);
   } while (assumed != old); // Repeat on failure/when the bin was already updated by another thread
}

///////////////////////////////////////////
/// Histogram filling kernels

/// @brief Fill the histogram using shared memory. Each block first fills a local histogram and then combines the
///        results of the local histogram into the final histogram stored in global memory to reduce atomic contention.
/// @param histogram Result array
/// @param binEdges Array with bin edges per axis, can be zero length if the bins are fixed size instead of variable
/// @param binEdgesIdx Index to the start of array for each axis in binEdges
/// @param nBinsAxis Number of bins per axis including under/overflow bins
/// @param xMin Minimum edge value per axis
/// @param xMax Maximum edge value per axis
/// @param coords Input coordinate values, in the form of xxx,yyy,zzz in case of multidimensional histograms
/// @param weights Weights that correspond to each coordinate
/// @param mask Mask for input coordinates for potential filtering TODO: implement this
/// @param nBins Total number of bins in the histogram
/// @param bulkSize Number of coordinates
template <typename T, unsigned int Dim>
__global__ void HistogramLocal(T *histogram, double *binEdges, int *binEdgesIdx, int *nBinsAxis, double *xMin,
                               double *xMax, double *coords, double *weights, bool *mask, size_t nBins, size_t bulkSize)
{
   auto sMem = CUDAHelpers::shared_memory_proxy<T>();
   unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
   unsigned int localTid = threadIdx.x;
   unsigned int stride = blockDim.x * gridDim.x; // total number of threads

   // Initialize a local per-block histogram
   for (auto i = localTid; i < nBins; i += blockDim.x) {
      sMem[i] = 0;
   }
   __syncthreads();

   // Fill local histogram
   for (auto i = tid; i < bulkSize; i += stride) {
      auto bin = GetBin<Dim>(i, binEdges, binEdgesIdx, nBinsAxis, xMin, xMax, coords, bulkSize, mask);
      // if (bin >= 0)
      AddBinContent<T>(sMem, bin, weights[i]);
   }
   __syncthreads();

   // Merge results in global histogram
   for (auto i = localTid; i < nBins; i += blockDim.x) {
      AddBinContent<T>(histogram, i, sMem[i]);
   }
}

// OPTIMIZATION: consider sorting the coords array.
/// @brief Fill the histogram using only global memory.
/// @param histogram Result array
/// @param binEdges Array with bin edges per axis, can be zero length if the bins are fixed size instead of variable
/// @param binEdgesIdx Index to the start of array for each axis in binEdges
/// @param nBinsAxis Number of bins per axis including under/overflow bins
/// @param xMin Minimum edge value per axis
/// @param xMax Maximum edge value per axis
/// @param coords Input coordinate values, in the form of xxx,yyy,zzz in case of multidimensional histograms
/// @param weights Weights that correspond to each coordinate
/// @param mask Mask for input coordinates for potential filtering TODO: implement this
/// @param bulkSize Number of coordinates
template <typename T, unsigned int Dim>
__global__ void HistogramGlobal(T *histogram, double *binEdges, int *binEdgesIdx, int *nBinsAxis, double *xMin,
                                double *xMax, double *coords, double *weights, bool *mask, size_t bulkSize)
{
   unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
   unsigned int stride = blockDim.x * gridDim.x;

   // Fill histogram
   for (auto i = tid; i < bulkSize; i += stride) {
      auto bin = GetBin<Dim>(i, binEdges, binEdgesIdx, nBinsAxis, xMin, xMax, coords, bulkSize, mask);
      // if (bin >= 0)
      AddBinContent<T>(histogram, bin, weights[i]);
   }
}

#define HISTOGRAM_GLOBAL(T, Dim) \
   template __global__ void HistogramGlobal<T, Dim>(                                    \
      T *histogram, double *binEdges, int *binEdgesIdx, int *nBinsAxis, double *xMin, \
      double *xMax, double *coords, double *weights, bool *mask, size_t bulkSize           \
   );

HISTOGRAM_GLOBAL(double, 1)
HISTOGRAM_GLOBAL(double, 2)
HISTOGRAM_GLOBAL(double, 3)

///////////////////////////////////////////
/// Statistics calculation kernels

/// @brief  Nullify weights of under/overflow mask to exclude them from stats
/// @param mask Mask over the coordinates
/// @param coords Weights for each coordinate
/// @param bulkSize Number of coordinates
template <unsigned int Dim, unsigned int BlockSize>
__global__ void ExcludeUOverflowKernel(bool *mask, double *weights, size_t bulkSize)
{
   unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
   unsigned int stride = blockDim.x * gridDim.x;

   for (auto i = tid; i < bulkSize; i += stride) {
      weights[i] *= mask[i];
   }
}

} // namespace Experimental
} // namespace ROOT
