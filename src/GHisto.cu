#include "CUDAHelpers.cuh"
#include "util.h"
#include "types.h"

#include "GHisto.h"

/// @brief Increase a bin in the histogram by a certain weight.
template <typename T>
__device__ inline void AddBinContent(T *histogram, u32 bin, f64 weight) {
    atomicAdd(&histogram[bin], (T)weight);
}

/// @brief Find the corresponding bin in a histogram axis based on a given x value.
__device__ inline u32 FindBin(f64 x, const f64 *binEdges, i32 binEdgesIdx, u32 nBins, f64 xMin, f64 xMax) {
    if (x < xMin)
        return 0;
    if (!(x < xMax))
        return nBins + 1;

    if (binEdgesIdx < 0)
        return 1 + u32(nBins * (x - xMin) / (xMax - xMin));
    return 1 + ROOT::Experimental::CUDAHelpers::BinarySearch(nBins + 1, &binEdges[binEdgesIdx], x);
}

/// @brief Calculate the corresponding bin for a value in an n-Dimensional histogram.
template<u32 Dim>
__device__ inline u32 GetBin(
    u32 i,
    f64 *binEdges, i32 *binEdgesIdx, u32 *nBinsAxis,
    f64 *xMin, f64 *xMax,
    f64 *coords, usize bulkSize
) {
    u32 bin = 0;
    for (int d = Dim - 1; d >= 0; --d) {
        f64 *x = &coords[d * bulkSize];
        u32 binD = FindBin(x[i], binEdges, binEdgesIdx[d], nBinsAxis[d] - 2, xMin[d], xMax[d]);
        bin = bin * nBinsAxis[d] + binD;
    }
    return bin;
}

/// @brief CUDA kernel to fill the histogram using global memory.
template <typename T, u32 Dim>
__global__ void HistogramGlobal(
    T* histogram,
    f64 *binEdges, i32 *binEdgesIdx, u32 *nBinsAxis,
    f64 *xMin, f64 *xMax,
    f64 *coords, f64 *weights, usize bulkSize
) {
    u32 tid = threadIdx.x + blockDim.x * blockIdx.x;
    u32 stride = blockDim.x * gridDim.x;

    for (auto i = tid; i < bulkSize; i += stride) {
        auto bin = GetBin<Dim>(i, binEdges, binEdgesIdx, nBinsAxis, xMin, xMax, coords, bulkSize);
        if (weights)
            AddBinContent<T>(histogram, bin, weights[i]);
        else
            AddBinContent<T>(histogram, bin, 1.0);
    }
}

template <typename T, u32 Dim, u32 BlockSize>
GHisto<T, Dim, BlockSize>::GHisto(
    const u32 *nBinsAxis,
    const f64 *xMin, const f64 *xMax,
    const f64 *binEdges, const i32 *binEdgesIdx,
    usize maxBulkSize
) {
    nBins = arrayProduct(nBinsAxis, Dim);
    this->maxBulkSize = maxBulkSize;

    ERRCHECK(cudaMalloc(&d_histogram, sizeof(T) * nBins));
    ERRCHECK(cudaMalloc(&d_nBinsAxis, sizeof(u32) * Dim));
    ERRCHECK(cudaMalloc(&d_xMin, sizeof(f64) * Dim));
    ERRCHECK(cudaMalloc(&d_xMax, sizeof(f64) * Dim));
    if (binEdges) {
        ERRCHECK(cudaMalloc(&d_binEdges, sizeof(f64) * arraySum(nBinsAxis, Dim)));
    } else {
        d_binEdges = nullptr;
    }
    ERRCHECK(cudaMalloc(&d_binEdgesIdx, sizeof(i32) * Dim));
    ERRCHECK(cudaMalloc(&d_coords, sizeof(f64) * Dim * maxBulkSize));
    ERRCHECK(cudaMalloc(&d_weights, sizeof(f64) * Dim * maxBulkSize));

    ERRCHECK(cudaMemset(d_histogram, 0, sizeof(T) * nBins));
    ERRCHECK(cudaMemcpy(d_nBinsAxis, nBinsAxis, sizeof(u32) * Dim, cudaMemcpyHostToDevice));
    ERRCHECK(cudaMemcpy(d_xMin, xMin, sizeof(f64) * Dim, cudaMemcpyHostToDevice));
    ERRCHECK(cudaMemcpy(d_xMax, xMax, sizeof(f64) * Dim, cudaMemcpyHostToDevice));
    if (d_binEdges) {
        ERRCHECK(cudaMemcpy(d_binEdges, binEdges, sizeof(f64) * arraySum(nBinsAxis, Dim), cudaMemcpyHostToDevice));
    }
    ERRCHECK(cudaMemcpy(d_binEdgesIdx, binEdgesIdx, sizeof(i32) * Dim, cudaMemcpyHostToDevice));
}

template <typename T, u32 Dim, u32 BlockSize>
GHisto<T, Dim, BlockSize>::~GHisto() {
    ERRCHECK(cudaFree(d_histogram));
    ERRCHECK(cudaFree(d_nBinsAxis));
    ERRCHECK(cudaFree(d_xMin));
    ERRCHECK(cudaFree(d_xMax));
    ERRCHECK(cudaFree(d_binEdges));
    ERRCHECK(cudaFree(d_binEdgesIdx));
    ERRCHECK(cudaFree(d_coords));
    ERRCHECK(cudaFree(d_weights));
}

template <typename T, u32 Dim, u32 BlockSize>
void GHisto<T, Dim, BlockSize>::RetrieveResults(T *histogram, f64 *stats) {
    ERRCHECK(cudaMemcpy(histogram, d_histogram, sizeof(T) * nBins, cudaMemcpyDeviceToHost));
}

template <typename T, u32 Dim, u32 BlockSize>
void GHisto<T, Dim, BlockSize>::Fill(u32 n, const f64 *coords) {
    if (n > maxBulkSize) {
        Fill(maxBulkSize, coords);
        Fill(n - maxBulkSize, coords + maxBulkSize);
        return;
    }

    ERRCHECK(cudaMemcpy(d_coords, coords, sizeof(f64) * n, cudaMemcpyHostToDevice));

    usize numBlocks = n % BlockSize == 0 ? n / BlockSize : n / BlockSize + 1;
    HistogramGlobal<T, Dim><<<numBlocks, BlockSize>>>(
        d_histogram,
        d_binEdges, d_binEdgesIdx, d_nBinsAxis,
        d_xMin, d_xMax,
        d_coords, nullptr, n
    );
    ERRCHECK(cudaPeekAtLastError());
}

template <typename T, u32 Dim, u32 BlockSize>
void GHisto<T, Dim, BlockSize>::Fill(u32 n, const f64 *coords, const f64 *weights) {
    if (n > maxBulkSize) {
        Fill(maxBulkSize, coords, weights);
        Fill(n - maxBulkSize, coords + maxBulkSize, weights + maxBulkSize);
        return;
    }

    ERRCHECK(cudaMemcpy(d_coords, coords, sizeof(f64) * n, cudaMemcpyHostToDevice));
    ERRCHECK(cudaMemcpy(d_weights, weights, sizeof(f64) * n, cudaMemcpyHostToDevice));

    usize numBlocks = n % BlockSize == 0 ? n / BlockSize : n / BlockSize + 1;
    HistogramGlobal<T, Dim><<<numBlocks, BlockSize>>>(
        d_histogram,
        d_binEdges, d_binEdgesIdx, d_nBinsAxis,
        d_xMin, d_xMax,
        d_coords, d_weights, n
    );
    ERRCHECK(cudaPeekAtLastError());
}

template class GHisto<double, 1>;
template class GHisto<double, 2>;
template class GHisto<double, 3>;
