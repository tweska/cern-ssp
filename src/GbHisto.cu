#include <iostream>

#include "CUDAHelpers.cuh"
#include "util.h"
#include "types.h"

#include "GbHisto.h"

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

/// @brief Find the corresponding bin in a histogram based on a coordinate.
__device__ inline u32 GetBin(
    u32 i, u32 nDims,
    f64 *binEdges, i32 *binEdgesIdx, u32 *nBinsAxis,
    f64 *xMin, f64 *xMax,
    f64 *coords, usize bulkSize
) {
    u32 bin = 0;
    for (int d = nDims - 1; d >= 0; --d) {
        f64 *x = &coords[d * bulkSize];
        u32 binD = FindBin(x[i], binEdges, binEdgesIdx[d], nBinsAxis[d] - 2, xMin[d], xMax[d]);
        bin = bin * nBinsAxis[d] + binD;
    }
    return bin;
}

/// @brief Global memory batch histogram kernel.
template <typename T>
__global__ void HistogramGlobal(
    T *histograms, u32 *histoResultOffset, u32 *histoOffset, u32 nHistos,
    f64 *binEdges, i32 *binEdgesOffset, u32 *nBinsAxis, u32 *nDims,
    f64 *xMin, f64 *xMax,
    f64 *coords, f64 *weights, usize bulkSize
) {
    u32 tid = threadIdx.x + blockDim.x * blockIdx.x;
    u32 stride = blockDim.x * gridDim.x;

    for (usize i = tid; i < bulkSize * nHistos; i += stride) {
        usize h = i / bulkSize;
        u32 hoff = histoOffset[h];  // Histogram Offset

        T *histogram = &histograms[histoResultOffset[h]];

        u32 bin = GetBin(
            i % bulkSize, nDims[h],
            &binEdges[hoff], &binEdgesOffset[hoff], &nBinsAxis[hoff],
            &xMin[hoff], &xMax[hoff],
            &coords[hoff * bulkSize], bulkSize
        );

        if (weights)
            AddBinContent<T>(histogram, bin, weights[i]);
        else
            AddBinContent<T>(histogram, bin, 1.0);
    }
}

template<typename T, u32 BlockSize>
GbHisto<T, BlockSize>::GbHisto(
    u32 nHistos, const u32 *nDims, const u32 *nBinsAxis,
    const f64 *xMin, const f64 *xMax,
    const f64 *binEdges, const i32 *binEdgesOffset,
    usize maxBulkSize
) {
    this->nHistos = nHistos;
    this->maxBulkSize = maxBulkSize;

    nBins = 0;
    nAxis = 0;
    h_histoResultOffset = new u32[nHistos];
    u32 h_histoOffset[nHistos];

    usize i = 0;
    for (usize h = 0; h < nHistos; ++h) {
        h_histoResultOffset[h] = nBins;
        nAxis += nDims[h];
        h_histoOffset[h] = i;
        u32 nInterBins = 1;
        for (usize d = 0; d < nDims[h]; ++d) {
            nInterBins *= nBinsAxis[i];
            i++;
        }
        nBins += nInterBins;
    }

    ERRCHECK(cudaMalloc(&d_histograms, sizeof(T) * nBins));
    if (binEdges) {
        ERRCHECK(cudaMalloc(&d_binEdges, sizeof(f64) * nBins));
    } else {
        d_binEdges = nullptr;
    }
    ERRCHECK(cudaMalloc(&d_xMin, sizeof(f64) * nAxis));
    ERRCHECK(cudaMalloc(&d_xMax, sizeof(f64) * nAxis));
    ERRCHECK(cudaMalloc(&d_nDims, sizeof(u32) * nHistos));
    ERRCHECK(cudaMalloc(&d_nBinsAxis, sizeof(u32) * nAxis));
    ERRCHECK(cudaMalloc(&d_histoResultOffset, sizeof(u32) * nHistos));
    ERRCHECK(cudaMalloc(&d_histoOffset, sizeof(u32) * nHistos));
    ERRCHECK(cudaMalloc(&d_binEdgesOffset, sizeof(i32) * nAxis));
    ERRCHECK(cudaMalloc(&d_coords, sizeof(f64) * nAxis * maxBulkSize));
    ERRCHECK(cudaMalloc(&d_weights, sizeof(f64) * nAxis * maxBulkSize));
    cudaDeviceSynchronize();

    ERRCHECK(cudaMemset(d_histograms, 0, sizeof(T) * nBins));
    if (d_binEdges) {
        ERRCHECK(cudaMemcpy(d_binEdges, binEdges, sizeof(f64) * nBins, cudaMemcpyHostToDevice));
    }
    ERRCHECK(cudaMemcpy(d_xMin, xMin, sizeof(f64) * nAxis, cudaMemcpyHostToDevice));
    ERRCHECK(cudaMemcpy(d_xMax, xMax, sizeof(f64) * nAxis, cudaMemcpyHostToDevice));
    ERRCHECK(cudaMemcpy(d_nDims, nDims, sizeof(u32) * nHistos, cudaMemcpyHostToDevice));
    ERRCHECK(cudaMemcpy(d_nBinsAxis, nBinsAxis, sizeof(i32) * nAxis, cudaMemcpyHostToDevice));
    ERRCHECK(cudaMemcpy(d_histoResultOffset, h_histoResultOffset, sizeof(u32) * nHistos, cudaMemcpyHostToDevice));
    ERRCHECK(cudaMemcpy(d_histoOffset, h_histoOffset, sizeof(u32) * nHistos, cudaMemcpyHostToDevice));
    ERRCHECK(cudaMemcpy(d_binEdgesOffset, binEdgesOffset, sizeof(i32) * nAxis, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
}

template<typename T, u32 BlockSize>
GbHisto<T, BlockSize>::~GbHisto() {
    ERRCHECK(cudaFree(d_histograms));
    ERRCHECK(cudaFree(d_binEdges));
    ERRCHECK(cudaFree(d_xMin));
    ERRCHECK(cudaFree(d_xMax));
    ERRCHECK(cudaFree(d_nDims));
    ERRCHECK(cudaFree(d_nBinsAxis));
    ERRCHECK(cudaFree(d_histoResultOffset));
    ERRCHECK(cudaFree(d_histoOffset));
    ERRCHECK(cudaFree(d_binEdgesOffset));
    ERRCHECK(cudaFree(d_coords));
    ERRCHECK(cudaFree(d_weights));
}

template <typename T, u32 BlockSize>
void GbHisto<T, BlockSize>::RetrieveResults(T *histograms, f64 *stats) {
    ERRCHECK(cudaMemcpy(histograms, d_histograms, sizeof(T) * nBins, cudaMemcpyDeviceToHost));
    ERRCHECK(cudaDeviceSynchronize());
}

template<typename T, u32 BlockSize>
void GbHisto<T, BlockSize>::Fill(u32 n, const f64 *coords) {
    Fill(n, coords, nullptr);
}

template<typename T, u32 BlockSize>
void GbHisto<T, BlockSize>::Fill(u32 n, const f64 *coords, const f64 *weights) {
    assert(n <= maxBulkSize);  // TODO: Split the bulk if the maxBulkSize is exceeded!

    ERRCHECK(cudaMemcpy(d_coords, coords, sizeof(f64) * nAxis * n, cudaMemcpyHostToDevice));

    f64 *weightsPtr = nullptr;
    if (weights) {
        ERRCHECK(cudaMemcpy(d_weights, weights, sizeof(f64) * n, cudaMemcpyHostToDevice));
        weightsPtr = d_weights;
    }

    usize nThreads = nHistos * n;
    usize nBlocks = nThreads / BlockSize + (nThreads % BlockSize != 0);

    HistogramGlobal<T><<<nBlocks, BlockSize>>>(
        d_histograms, d_histoResultOffset, d_histoOffset, nHistos,
        d_binEdges, d_binEdgesOffset, d_nBinsAxis, d_nDims,
        d_xMin, d_xMax,
        d_coords, weightsPtr, n
    );
    ERRCHECK(cudaPeekAtLastError());
}


template class GbHisto<double>;