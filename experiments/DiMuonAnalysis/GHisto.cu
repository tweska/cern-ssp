#include "../../inc/types.h"
#include "../../inc/CUDAHelpers.cuh"

#include "GHisto.h"

/// @brief Return the invariant mass of two particles given their
/// transverse momentum (pt), rapidity (eta), azimuth (phi) and mass.
__device__
f32 inline invariantMass(
    f32 pt0, f32 eta0, f32 phi0, f32 mass0,
    f32 pt1, f32 eta1, f32 phi1, f32 mass1
) {
    f32 x_sum = 0.0;
    f32 y_sum = 0.0;
    f32 z_sum = 0.0;
    f32 e_sum = 0.0;

    {
        const auto x = pt0 * cos(phi0);
        x_sum += x;
        const auto y = pt0 * sin(phi0);
        y_sum += y;
        const auto z = pt0 * sinh(eta0);
        z_sum += z;
        const auto e = sqrt(x * x + y * y + z * z + mass0 * mass0);
        e_sum += e;
    } {
        const auto x = pt1 * cos(phi1);
        x_sum += x;
        const auto y = pt1 * sin(phi1);
        y_sum += y;
        const auto z = pt1 * sinh(eta1);
        z_sum += z;
        const auto e = sqrt(x * x + y * y + z * z + mass1 * mass1);
        e_sum += e;
    }
    return sqrt(e_sum * e_sum - x_sum * x_sum - y_sum * y_sum - z_sum * z_sum);
}

/// @brief Return the invariant mass of two particles given their
/// transverse momentum (pt), rapidity (eta), azimuth (phi) and mass.
__device__
f64 invariantCoordMass(f64 *coords, usize n)
{
    return invariantMass(
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

template class GHisto<f64, invariantCoordMass>;
