#include "../../inc/types.h"
#include "../../inc/CUDAHelpers.cuh"

#include "GHisto.h"

__device__
f32 inline Angle(f32 x0, f32 y0, f32 z0, f32 x1, f32 y1, f32 z1) {
    // cross product
    const auto cx = y0 * z1 - y1 * z0;
    const auto cy = x0 * z1 - x1 * z0;
    const auto cz = x0 * y1 - x1 * y0;

    // norm of cross product
    const auto c = std::sqrt(cx * cx + cy * cy + cz * cz);

    // dot product
    const auto  d = x0 * x1 + y0 * y1 + z0 * z1;

    return atan2(c, d);
}

__device__
f32 InvariantMassesPxPyPzM(
   const f32 x0, const f32 y0, const f32 z0, const f32 mass0,
   const f32 x1, const f32 y1, const f32 z1, const f32 mass1
) {
    // Numerically stable computation of Invariant Masses
    const auto p0_sq = x0 * x0 + y0 * y0 + z0 * z0;
    const auto p1_sq = x1 * x1 + y1 * y1 + z1 * z1;

    if (p0_sq <= 0 && p1_sq <= 0)
        return (mass0 + mass1);
    if (p0_sq <= 0) {
        auto mm = mass0 + sqrt(mass1*mass1 + p1_sq);
        auto m2 = mm*mm - p1_sq;
        return m2 >= 0 ? sqrt(m2) : sqrt(-m2);
    }
    if (p1_sq <= 0) {
        auto mm = mass1 + sqrt(mass0*mass0 + p0_sq);
        auto m2 = mm*mm - p0_sq;
        return m2 >= 0 ? sqrt(m2) : sqrt(-m2);
    }

    const auto m0_sq =  mass0 * mass0;
    const auto m1_sq =  mass1 * mass1;

    const auto r0 = m0_sq / p0_sq;
    const auto r1 = m1_sq / p1_sq;
    const auto x = r0 + r1 + r0 * r1;
    const auto a = Angle(x0, y0, z0, x1, y1, z1);
    const auto cos_a = cos(a);
    auto y = x;
    if (cos_a >= 0){
        y = (x + sin(a) * sin(a)) / (sqrt(x + 1) + cos_a);
    } else {
        y = sqrt(x + 1) - cos_a;
    }

    const auto z = 2 * sqrt(p0_sq * p1_sq);

    // Return invariant mass with (+, -, -, -) metric
    return sqrt(m0_sq + m1_sq + y * z);
}


/// @brief Return the invariant mass of two particles given their
/// transverse momentum (pt), rapidity (eta), azimuth (phi) and mass.
__device__
f32 inline InvariantMasses(
    f32 pt0, f32 eta0, f32 phi0, f32 mass0,
    f32 pt1, f32 eta1, f32 phi1, f32 mass1
) {
    const auto x0 = pt0 * cos(phi0);
    const auto y0 = pt0 * sin(phi0);
    const auto z0 = pt0 * sinh(eta0);

    const auto x1 = pt1 * cos(phi1);
    const auto y1 = pt1 * sin(phi1);
    const auto z1 = pt1 * sinh(eta1);

    return InvariantMassesPxPyPzM(x0, y0, z0, mass0, x1, y1, z1, mass1);
}

/// @brief Return the invariant mass of two particles given their
/// transverse momentum (pt), rapidity (eta), azimuth (phi) and mass.
__device__
f64 InvariantCoordMasses(f64 *coords, usize n)
{
    return InvariantMasses(
        coords[0 * n], coords[2 * n], coords[4 * n], coords[6 * n],
        coords[1 * n], coords[3 * n], coords[5 * n], coords[7 * n]
    );
}

/// @brief Increase a bin in the histogram by a certain weight.
template <typename T>
__device__ inline void AddBinContent(T *histogram, usize bin, f64 weight) {
    atomicAdd(&histogram[bin], (T)weight);
}

/// @brief Find the corresponding bin in a histogram axis based on a given x value.
__device__ inline usize FindBin(f64 x, usize nBins, f64 xMin, f64 xMax) {
    if (x < xMin)
        return 0;
    if (!(x < xMax))
        return nBins + 1;

    return 1 + usize(nBins * (x - xMin) / (xMax - xMin));
}

/// @brief Calculate the corresponding bin for a value in an n-Dimensional histogram.
template<Op op>
__device__ inline usize GetBin(
    usize i,
    usize nBins,
    f64 xMin, f64 xMax,
    f64 *coords, usize bulkSize
) {
    const f64 x = op(&coords[i], bulkSize);
    return FindBin(x, nBins - 2, xMin, xMax);
}

/// @brief CUDA kernel to fill the histogram using global memory.
template <typename T, Op op>
__global__ void HistogramKernel(
    T* histogram, usize nBins,
    f64 xMin, f64 xMax,
    f64 *coords, f64 *weights, usize bulkSize
) {
    usize tid = threadIdx.x + blockDim.x * blockIdx.x;
    usize stride = blockDim.x * gridDim.x;

    for (auto i = tid; i < bulkSize; i += stride) {
        auto bin = GetBin<op>(i, nBins, xMin, xMax, coords, bulkSize);
        if (weights)
            AddBinContent<T>(histogram, bin, weights[i]);
        else
            AddBinContent<T>(histogram, bin, 1.0);
    }
}

template<typename T, Op * op, usize BlockSize>
GHisto<T, op, BlockSize>::GHisto(
    usize nBins,
    f64 xMin, f64 xMax,
    usize bulkDims, usize maxBulkSize
) {
    this->nBins = nBins;
    this->xMin = xMin;
    this->xMax = xMax;
    this->bulkDims = bulkDims;
    this->maxBulkSize = maxBulkSize;

    ERRCHECK(cudaMalloc(&d_histogram, sizeof(T) * nBins));
    ERRCHECK(cudaMalloc(&d_coords, sizeof(f64) * bulkDims * maxBulkSize));
    ERRCHECK(cudaMalloc(&d_weights, sizeof(f64) * maxBulkSize));
}

template<typename T, Op * op, usize BlockSize>
GHisto<T, op, BlockSize>::~GHisto() {
    ERRCHECK(cudaFree(d_histogram));
    ERRCHECK(cudaFree(d_coords));
    ERRCHECK(cudaFree(d_weights));
}

template<typename T, Op * op, usize BlockSize>
void GHisto<T, op, BlockSize>::RetrieveResults(T *histogram) {
    ERRCHECK(cudaMemcpy(histogram, d_histogram, sizeof(T) * nBins, cudaMemcpyDeviceToHost));
}

template<typename T, Op * op, usize BlockSize>
void GHisto<T, op, BlockSize>::FillN(usize n, const f64 *coords, const f64 *weights) {
    assert(n <= maxBulkSize);

    ERRCHECK(cudaMemcpy(d_coords, coords, sizeof(f64) * bulkDims * n, cudaMemcpyHostToDevice));

    f64 *weightsPtr = nullptr;
    if (weights) {
        ERRCHECK(cudaMemcpy(d_weights, weights, sizeof(f64) * n, cudaMemcpyHostToDevice));
        weightsPtr = d_weights;
    }

    usize numBlocks = n % BlockSize == 0 ? n / BlockSize : n / BlockSize + 1;
    HistogramKernel<T, op><<<numBlocks, BlockSize>>>(
        d_histogram, nBins,
        xMin, xMax,
        d_coords, weightsPtr, n
    );
    ERRCHECK(cudaPeekAtLastError());
}

template class GHisto<f64, InvariantCoordMasses>;
