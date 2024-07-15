#ifndef RHnCUDA_H
#define RHnCUDA_H

namespace ROOT {
namespace Experimental {

template <typename T, unsigned int Dim>
__global__ void HistogramLocal(T *histogram, double *binEdges, int *binEdgesIdx, int *nBinsAxis, double *xMin,
                               double *xMax, double *coords, double *weights, bool *mask, size_t nBins, size_t bulkSize);

template <typename T, unsigned int Dim>
__global__ void HistogramGlobal(T *histogram, double *binEdges, int *binEdgesIdx, int *nBinsAxis, double *xMin,
                                double *xMax, double *coords, double *weights, bool *mask, size_t bulkSize);

template <unsigned int Dim, unsigned int BlockSize>
__global__ void ExcludeUOverflowKernel(bool *mask, double *weights, size_t bulkSize);

} // namespace Experimental
} // namespace ROOT
#endif
