#include "CUDAHelpers.cuh"
#include "util.h"
#include "types.h"

#include "GHisto.h"

/// @brief Increase a bin in the histogram by a certain weight.
template <typename T>
__device__ inline void AddBinContent(T *histogram, usize bin, f64 weight) {
    atomicAdd(&histogram[bin], (T)weight);
}

/// @brief Find the corresponding bin in a histogram axis based on a given x value.
__device__ inline usize FindBin(f64 x, const f64 *binEdges, isize binEdgesIdx, usize nBins, f64 xMin, f64 xMax) {
    if (x < xMin)
        return 0;
    if (!(x < xMax))
        return nBins + 1;

    if (binEdgesIdx < 0)
        return 1 + usize(nBins * (x - xMin) / (xMax - xMin));
    return 1 + ROOT::Experimental::CUDAHelpers::BinarySearch(nBins + 1, &binEdges[binEdgesIdx], x);
}

/// @brief Calculate the corresponding bin for a value in an n-Dimensional histogram.
template<usize Dim>
__device__ inline usize GetBin(
    usize i,
    f64 *binEdges, isize *binEdgesIdx, usize *nBinsAxis,
    f64 *xMin, f64 *xMax,
    f64 *coords, usize bulkSize
) {
    usize bin = 0;
    for (isize d = Dim - 1; d >= 0; --d) {
        f64 *x = &coords[d * bulkSize];
        usize binD = FindBin(x[i], binEdges, binEdgesIdx[d], nBinsAxis[d] - 2, xMin[d], xMax[d]);
        bin = bin * nBinsAxis[d] + binD;
    }
    return bin;
}

/// @brief CUDA kernel to fill the histogram using global memory.
template <typename T, usize Dim>
__global__ void HistogramGlobal(
    T* histogram,
    f64 *binEdges, isize *binEdgesIdx, usize *nBinsAxis,
    f64 *xMin, f64 *xMax,
    f64 *coords, f64 *weights, usize bulkSize
) {
    usize tid = threadIdx.x + blockDim.x * blockIdx.x;
    usize stride = blockDim.x * gridDim.x;

    for (auto i = tid; i < bulkSize; i += stride) {
        auto bin = GetBin<Dim>(i, binEdges, binEdgesIdx, nBinsAxis, xMin, xMax, coords, bulkSize);
        if (weights)
            AddBinContent<T>(histogram, bin, weights[i]);
        else
            AddBinContent<T>(histogram, bin, 1.0);
    }
}

template <typename T, usize Dim, usize BlockSize>
GHisto<T, Dim, BlockSize>::GHisto(
    const usize *nBinsAxis,
    const f64 *xMin, const f64 *xMax,
    const f64 *binEdges, const isize *binEdgesIdx,
    usize maxBulkSize
) {
    nBins = arrayProduct(nBinsAxis, Dim);
    this->maxBulkSize = maxBulkSize;

    ERRCHECK(cudaMalloc(&d_histogram, sizeof(T) * nBins));
    ERRCHECK(cudaMalloc(&d_nBinsAxis, sizeof(usize) * Dim));
    ERRCHECK(cudaMalloc(&d_xMin, sizeof(f64) * Dim));
    ERRCHECK(cudaMalloc(&d_xMax, sizeof(f64) * Dim));
    if (binEdges) {
        ERRCHECK(cudaMalloc(&d_binEdges, sizeof(f64) * arraySum(nBinsAxis, Dim)));
    } else {
        d_binEdges = nullptr;
    }
    ERRCHECK(cudaMalloc(&d_binEdgesIdx, sizeof(isize) * Dim));
    ERRCHECK(cudaMalloc(&d_coords, sizeof(f64) * Dim * maxBulkSize));
    ERRCHECK(cudaMalloc(&d_weights, sizeof(f64) * maxBulkSize));

    ERRCHECK(cudaMemset(d_histogram, 0, sizeof(T) * nBins));
    ERRCHECK(cudaMemcpy(d_nBinsAxis, nBinsAxis, sizeof(usize) * Dim, cudaMemcpyHostToDevice));
    ERRCHECK(cudaMemcpy(d_xMin, xMin, sizeof(f64) * Dim, cudaMemcpyHostToDevice));
    ERRCHECK(cudaMemcpy(d_xMax, xMax, sizeof(f64) * Dim, cudaMemcpyHostToDevice));
    if (d_binEdges) {
        ERRCHECK(cudaMemcpy(d_binEdges, binEdges, sizeof(f64) * arraySum(nBinsAxis, Dim), cudaMemcpyHostToDevice));
    }
    ERRCHECK(cudaMemcpy(d_binEdgesIdx, binEdgesIdx, sizeof(isize) * Dim, cudaMemcpyHostToDevice));
}

template <typename T, usize Dim, usize BlockSize>
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

template <typename T, usize Dim, usize BlockSize>
void GHisto<T, Dim, BlockSize>::RetrieveResults(T *histogram) {
    ERRCHECK(cudaMemcpy(histogram, d_histogram, sizeof(T) * nBins, cudaMemcpyDeviceToHost));
}

template <typename T, usize Dim, usize BlockSize>
void GHisto<T, Dim, BlockSize>::Fill(usize n, const f64 *coords) {
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

template <typename T, usize Dim, usize BlockSize>
void GHisto<T, Dim, BlockSize>::Fill(usize n, const f64 *coords, const f64 *weights) {
    if (n > maxBulkSize) {
        Fill(maxBulkSize, coords, weights);
        Fill(n - maxBulkSize, coords + maxBulkSize, weights + maxBulkSize);
        return;
    }

    ERRCHECK(cudaMemcpy(d_coords, coords, sizeof(f64) * Dim * n, cudaMemcpyHostToDevice));

    f64 *weightsPtr = nullptr;
    if (weights) {
        ERRCHECK(cudaMemcpy(d_weights, weights, sizeof(f64) * n, cudaMemcpyHostToDevice));
        weightsPtr = d_weights;
    }

    usize numBlocks = n % BlockSize == 0 ? n / BlockSize : n / BlockSize + 1;
    HistogramGlobal<T, Dim><<<numBlocks, BlockSize>>>(
        d_histogram,
        d_binEdges, d_binEdgesIdx, d_nBinsAxis,
        d_xMin, d_xMax,
        d_coords, weightsPtr, n
    );
    ERRCHECK(cudaPeekAtLastError());
}

template class GHisto<f64, 1>;
template class GHisto<f64, 2>;
template class GHisto<f64, 3>;
