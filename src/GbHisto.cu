#include <chrono>
#include <iostream>
#include <iomanip>

#include "CUDAHelpers.cuh"
#include "types.h"

#include "GbHisto.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

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

/// @brief Find the corresponding bin in a histogram based on a coordinate.
__device__ inline usize GetBin(
    usize i, usize nDims,
    f64 *binEdges, isize *binEdgesIdx, usize *nBinsAxis,
    f64 *xMin, f64 *xMax,
    f64 *coords, usize bulkSize
) {
    usize bin = 0;
    for (isize d = nDims - 1; d >= 0; --d) {
        f64 *x = &coords[d * bulkSize];
        usize binD = FindBin(x[i], binEdges, binEdgesIdx[d], nBinsAxis[d] - 2, xMin[d], xMax[d]);
        bin = bin * nBinsAxis[d] + binD;
    }
    return bin;
}

/// @brief Global memory batch histogram kernel.
template <typename T>
__global__ void HistogramGlobal(
    T *histograms, usize *histoResultOffset, usize *histoOffset, usize nHistos,
    f64 *binEdges, isize *binEdgesOffset, usize *nBinsAxis, usize *nDims,
    f64 *xMin, f64 *xMax,
    f64 *coords, f64 *weights, usize bulkSize
) {
    usize tid = threadIdx.x + blockDim.x * blockIdx.x;
    usize stride = blockDim.x * gridDim.x;

    for (usize i = tid; i < bulkSize * nHistos; i += stride) {
        usize h = i / bulkSize;
        usize hoff = histoOffset[h];  // Histogram Offset

        T *histogram = &histograms[histoResultOffset[h]];

        usize bin = GetBin(
            i % bulkSize, nDims[h],
            binEdges, &binEdgesOffset[hoff], &nBinsAxis[hoff],
            &xMin[hoff], &xMax[hoff],
            &coords[hoff * bulkSize], bulkSize
        );

        if (weights)
            AddBinContent<T>(histogram, bin, weights[i]);
        else
            AddBinContent<T>(histogram, bin, 1.0);
    }
}

template<typename T, usize BlockSize>
GbHisto<T, BlockSize>::GbHisto(
    usize nHistos, const usize *nDims, const usize *nBinsAxis,
    const f64 *xMin, const f64 *xMax,
    const f64 *binEdges, const isize *binEdgesOffset,
    usize maxBulkSize
) {
    auto t0 = high_resolution_clock::now();

    this->nHistos = nHistos;
    this->maxBulkSize = maxBulkSize;

    nBins = 0;
    nAxis = 0;
    h_histoResultOffset = new usize[nHistos];
    h_histoOffset = new usize[nHistos];

    for (usize h = 0, i = 0; h < nHistos; ++h) {
        h_histoResultOffset[h] = nBins;
        nAxis += nDims[h];
        h_histoOffset[h] = i;
        usize nInterBins = 1;
        for (usize d = 0; d < nDims[h]; ++d, ++i) {
            nInterBins *= nBinsAxis[i];
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
    ERRCHECK(cudaMalloc(&d_nDims, sizeof(usize) * nHistos));
    ERRCHECK(cudaMalloc(&d_nBinsAxis, sizeof(usize) * nAxis));
    ERRCHECK(cudaMalloc(&d_histoResultOffset, sizeof(usize) * nHistos));
    ERRCHECK(cudaMalloc(&d_histoOffset, sizeof(usize) * nHistos));
    ERRCHECK(cudaMalloc(&d_binEdgesOffset, sizeof(isize) * nAxis));
    ERRCHECK(cudaMalloc(&d_coords, sizeof(f64) * nAxis * maxBulkSize));
    ERRCHECK(cudaMalloc(&d_weights, sizeof(f64) * nHistos * maxBulkSize));
    cudaDeviceSynchronize();

    ERRCHECK(cudaMemset(d_histograms, 0, sizeof(T) * nBins));
    if (d_binEdges) {
        ERRCHECK(cudaMemcpy(d_binEdges, binEdges, sizeof(f64) * nBins, cudaMemcpyHostToDevice));
    }
    ERRCHECK(cudaMemcpy(d_xMin, xMin, sizeof(f64) * nAxis, cudaMemcpyHostToDevice));
    ERRCHECK(cudaMemcpy(d_xMax, xMax, sizeof(f64) * nAxis, cudaMemcpyHostToDevice));
    ERRCHECK(cudaMemcpy(d_nDims, nDims, sizeof(usize) * nHistos, cudaMemcpyHostToDevice));
    ERRCHECK(cudaMemcpy(d_nBinsAxis, nBinsAxis, sizeof(usize) * nAxis, cudaMemcpyHostToDevice));
    ERRCHECK(cudaMemcpy(d_histoResultOffset, h_histoResultOffset, sizeof(usize) * nHistos, cudaMemcpyHostToDevice));
    ERRCHECK(cudaMemcpy(d_histoOffset, h_histoOffset, sizeof(usize) * nHistos, cudaMemcpyHostToDevice));
    ERRCHECK(cudaMemcpy(d_binEdgesOffset, binEdgesOffset, sizeof(isize) * nAxis, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    auto t1 = high_resolution_clock::now();
    duration<f64, std::milli> runtimeMs = t1 - t0;
    runtimeInit = runtimeMs.count();
}

template<typename T, usize BlockSize>
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

    delete[] h_histoResultOffset;
    delete[] h_histoOffset;
}

template <typename T, usize BlockSize>
void GbHisto<T, BlockSize>::RetrieveResults(T *histograms, f64 *stats) {
    auto t0 = high_resolution_clock::now();

    ERRCHECK(cudaMemcpy(histograms, d_histograms, sizeof(T) * nBins, cudaMemcpyDeviceToHost));
    ERRCHECK(cudaDeviceSynchronize());

    auto t1 = high_resolution_clock::now();
    duration<f64, std::milli> runtimeMs = t1 - t0;
    runtimeResult = runtimeMs.count();
}

template <typename T, usize BlockSize>
void GbHisto<T, BlockSize>::GetRuntime(f64 *runtimeInit, f64 *runtimeTransfer, f64 *runtimeKernel, f64 *runtimeResult) {
    *runtimeInit = this->runtimeInit;
    *runtimeTransfer = this->runtimeTransfer;
    *runtimeKernel = this->runtimeKernel;
    *runtimeResult = this->runtimeResult;
}

template <typename T, usize BlockSize>
void GbHisto<T, BlockSize>::PrintRuntime(std::ostream &output) {
    output << "runtimeInit     = " << std::setw(10) << std::fixed << std::setprecision(5) << runtimeInit << std::endl;
    output << "runtimeTransfer = " << std::setw(10) << std::fixed << std::setprecision(5) << runtimeTransfer << std::endl;
    output << "runtimeKernel   = " << std::setw(10) << std::fixed << std::setprecision(5) << runtimeKernel << std::endl;
    output << "runtimeResult   = " << std::setw(10) << std::fixed << std::setprecision(5) << runtimeResult << std::endl;
    output << "runtimeFill     = " << std::setw(10) << std::fixed << std::setprecision(5) << runtimeTransfer + runtimeKernel << "  (coords + kernel)" << std::endl;
    output << "runtimeTotal    = " << std::setw(10) << std::fixed << std::setprecision(5) << runtimeInit + runtimeTransfer + runtimeKernel + runtimeResult << "  (init + coords + kernel + result)" << std::endl;
}

template<typename T, usize BlockSize>
void GbHisto<T, BlockSize>::Fill(usize n, const f64 *coords) {
    Fill(n, coords, nullptr);
}

template<typename T, usize BlockSize>
void GbHisto<T, BlockSize>::Fill(usize n, const f64 *coords, const f64 *weights) {
    if (n > maxBulkSize) {
        Fill(maxBulkSize, coords, weights);
        Fill(n - maxBulkSize, coords + maxBulkSize, weights + maxBulkSize);
        return;
    }

    auto t0 = high_resolution_clock::now();

    ERRCHECK(cudaMemcpy(d_coords, coords, sizeof(f64) * nAxis * n, cudaMemcpyHostToDevice));

    f64 *weightsPtr = nullptr;
    if (weights) {
        ERRCHECK(cudaMemcpy(d_weights, weights, sizeof(f64) * n * nHistos, cudaMemcpyHostToDevice));
        weightsPtr = d_weights;
    }

    auto t1 = high_resolution_clock::now();

    usize nThreads = nHistos * n;
    usize nBlocks = nThreads / BlockSize + (nThreads % BlockSize != 0);

    HistogramGlobal<T><<<nBlocks, BlockSize>>>(
        d_histograms, d_histoResultOffset, d_histoOffset, nHistos,
        d_binEdges, d_binEdgesOffset, d_nBinsAxis, d_nDims,
        d_xMin, d_xMax,
        d_coords, weightsPtr, n
    );
    ERRCHECK(cudaPeekAtLastError());

    auto t2 = high_resolution_clock::now();
    duration<f64, std::milli> runtimeMsCoords = t1 - t0;
    duration<f64, std::milli> runtimeMsKernel = t2 - t1;
    runtimeTransfer += runtimeMsCoords.count();
    runtimeKernel += runtimeMsKernel.count();
}


template class GbHisto<f64>;
