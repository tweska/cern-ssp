#include "CUDAHelpers.cuh"
#include "types.h"

#include "GpuDefHisto.h"

/// @brief Increase a bin in the histogram by a certain weight.
template <typename T>
__device__ inline void AddBinContent(T *histogram, usize bin, f64 weight)
{
    atomicAdd(&histogram[bin], (T)weight);
}

/// @brief Find the corresponding bin in a histogram axis based on a given x value.
__device__ inline usize FindBin(f64 x, usize nBins, f64 xMin, f64 xMax)
{
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
    const f64 x = op(coords, i, bulkSize);
    return FindBin(x, nBins - 2, xMin, xMax);
}

/// @brief CUDA kernel to fill the histogram using global memory.
template<typename T, Op op>
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

/// @brief CUDA kernel to fill the histogram using global memory.
template<Op op>
__global__ void ExecuteOpKernel(
    usize *bin, f64 *coord,
    usize nBins, f64 xMin, f64 xMax
) {
    usize tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid == 0) {
        *bin = GetBin<op>(0, nBins, xMin, xMax, coord, 1);
    }
}

template<typename T, Op *op, usize BlockSize>
GpuDefHisto<T, op, BlockSize>::GpuDefHisto(
    usize nBins,
    f64 xMin, f64 xMax,
    usize bulkDims, usize maxBulkSize,
    Timer<> *timerTransfer, Timer<> *timerFill, Timer<> *timerResult
) {
    this->nBins = nBins;
    this->xMin = xMin;
    this->xMax = xMax;
    this->bulkDims = bulkDims;
    this->maxBulkSize = maxBulkSize;

    ERRCHECK(cudaMalloc(&d_histogram, sizeof(T) * nBins));
    ERRCHECK(cudaMalloc(&d_coords, sizeof(f64) * bulkDims * maxBulkSize));
    ERRCHECK(cudaMalloc(&d_weights, sizeof(f64) * maxBulkSize));
    ERRCHECK(cudaDeviceSynchronize());

    this->timerTransfer = timerTransfer;
    this->timerFill = timerFill;
    this->timerResult = timerResult;
}

template<typename T, Op *op, usize BlockSize>
GpuDefHisto<T, op, BlockSize>::GpuDefHisto(
    usize nBins,
    f64 xMin, f64 xMax,
    usize bulkDims, usize maxBulkSize
    ) : GpuDefHisto(
        nBins,
        xMin, xMax,
        bulkDims, maxBulkSize,
        nullptr, nullptr, nullptr
    ) {}

template<typename T, Op *op, usize BlockSize>
GpuDefHisto<T, op, BlockSize>::~GpuDefHisto()
{
    ERRCHECK(cudaFree(d_histogram));
    ERRCHECK(cudaFree(d_coords));
    ERRCHECK(cudaFree(d_weights));
}

template<typename T, Op *op, usize BlockSize>
void GpuDefHisto<T, op, BlockSize>::RetrieveResults(T *histogram)
{
    if (timerResult) timerResult->start();
    ERRCHECK(cudaMemcpy(histogram, d_histogram, sizeof(T) * nBins, cudaMemcpyDeviceToHost));
    ERRCHECK(cudaDeviceSynchronize());
    if (timerResult) timerResult->pause();
}

template<typename T, Op *op, usize BlockSize>
void GpuDefHisto<T, op, BlockSize>::FillN(
    usize n, const f64 *coords, const f64 *weights
) {
    assert(n <= maxBulkSize);

    if (timerTransfer) timerTransfer->start();

    ERRCHECK(cudaMemcpy(d_coords, coords, sizeof(f64) * bulkDims * n, cudaMemcpyHostToDevice));

    f64 *weightsPtr = nullptr;
    if (weights) {
        ERRCHECK(cudaMemcpy(d_weights, weights, sizeof(f64) * n, cudaMemcpyHostToDevice));
        weightsPtr = d_weights;
    }
    ERRCHECK(cudaDeviceSynchronize());

    if (timerTransfer) timerTransfer->pause();
    if (timerFill) timerFill->start();

    usize numBlocks = n % BlockSize == 0 ? n / BlockSize : n / BlockSize + 1;
    HistogramKernel<T, op><<<numBlocks, BlockSize>>>(
        d_histogram, nBins,
        xMin, xMax,
        d_coords, weightsPtr, n
    );
    ERRCHECK(cudaDeviceSynchronize());
    ERRCHECK(cudaPeekAtLastError());

    if (timerFill) timerFill->pause();
}

template<typename T, Op *op, usize BlockSize>
usize GpuDefHisto<T, op, BlockSize>::ExecuteOp(const f64 *coord)
{
    usize h_bin, *d_bin;
    ERRCHECK(cudaMalloc(&d_bin, sizeof(usize)));
    ERRCHECK(cudaMemcpy(d_coords, coord, sizeof(f64) * bulkDims, cudaMemcpyHostToDevice));

    ExecuteOpKernel<op><<<1, BlockSize>>>(d_bin, d_coords, nBins, xMin, xMax);
    ERRCHECK(cudaDeviceSynchronize());
    ERRCHECK(cudaPeekAtLastError());

    ERRCHECK(cudaMemcpy(&h_bin, d_bin, sizeof(usize), cudaMemcpyDeviceToHost));
    ERRCHECK(cudaDeviceSynchronize());
    ERRCHECK(cudaFree(d_bin));

    return h_bin;
}
