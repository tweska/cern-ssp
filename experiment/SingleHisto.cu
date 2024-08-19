#include <random>

// ROOT Histogramming
#include "TH1.h"

#include "CUDAHelpers.cuh"
#include "util.h"
#include "types.h"

typedef f64 (Op)(f64*, usize);

__device__
double add2(double *x, usize bulkSize) {
    return x[0 * bulkSize] + x[1 * bulkSize];
}

__device__
void add2(double *x, double *result, usize bulkSize) {
    *result = x[0 * bulkSize] + x[1 * bulkSize];
}

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
template<usize Dim, Op op>
__device__ inline usize GetBin(
    usize i,
    f64 *binEdges, isize *binEdgesIdx, usize *nBinsAxis,
    f64 *xMin, f64 *xMax,
    f64 *coords, usize bulkSize
) {
    usize bin = 0;
    for (int d = Dim - 1; d >= 0; --d) {
        f64 x = op(&coords[i], bulkSize);
        usize binD = FindBin(x, binEdges, binEdgesIdx[d], nBinsAxis[d] - 2, xMin[d], xMax[d]);
        bin = bin * nBinsAxis[d] + binD;
    }
    return bin;
}

/// @brief CUDA kernel to fill the histogram using global memory.
template <typename T, usize Dim, Op op>
__global__ void HistogramGlobal(
    T* histogram,
    f64 *binEdges, isize *binEdgesIdx, usize *nBinsAxis,
    f64 *xMin, f64 *xMax,
    f64 *coords, f64 *weights, usize bulkDim, usize bulkSize
) {
    usize tid = threadIdx.x + blockDim.x * blockIdx.x;
    usize stride = blockDim.x * gridDim.x;

    for (auto i = tid; i < bulkSize; i += stride) {
        auto bin = GetBin<Dim, op>(i, binEdges, binEdgesIdx, nBinsAxis, xMin, xMax, coords, bulkSize);
        if (weights)
            AddBinContent<T>(histogram, bin, weights[i]);
        else
            AddBinContent<T>(histogram, bin, 1.0);
    }
}

/// @brief CUDA histogramming class
/// @tparam T Histogram data type
/// @tparam Dim Dimensionality of the histogram
/// @tparam BlockSize Cuda block size to use in the kernels
template <typename T, Op op, usize Dim, usize BlockSize = 256>
class SingleHisto {
private:
    T     *d_histogram;   ///< Histogram buffer
    usize    nBins;       ///< Total number of bins in the histogram (with under/overflow)

    usize *d_nBinsAxis;   ///< Number of bins per axis (with under/overflow)
    f64   *d_xMin;        ///< Lower edge of first bin per axis
    f64   *d_xMax;        ///< Upper edge of last bin per axis
    f64   *d_binEdges;    ///< Bin edges array for each axis (may be null for fixed bins)
    isize *d_binEdgesIdx; ///< Start index of the binedges in binEdges (-1 for fixed bins)

    f64 *d_coords;        ///< Coordinates buffer
    f64 *d_weights;       ///< Weights buffer

    usize  bulkDim;       ///< Number of dimensions in the coordinate buffer
    usize  maxBulkSize;   ///< Size of the coordinates buffer

public:
    SingleHisto() = delete;

    SingleHisto(
        const usize *nBinsAxis,
        const f64 *xMin, const f64 *xMax,
        const f64 *binEdges, const isize *binEdgesIdx,
        usize bulkDim, usize maxBulkSize
    ) {
        nBins = arrayProduct(nBinsAxis, Dim);
        this->bulkDim = bulkDim;
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
        ERRCHECK(cudaMalloc(&d_coords, sizeof(f64) * bulkDim * maxBulkSize));
        ERRCHECK(cudaMalloc(&d_weights, sizeof(f64) * Dim * maxBulkSize));

        ERRCHECK(cudaMemset(d_histogram, 0, sizeof(T) * nBins));
        ERRCHECK(cudaMemcpy(d_nBinsAxis, nBinsAxis, sizeof(usize) * Dim, cudaMemcpyHostToDevice));
        ERRCHECK(cudaMemcpy(d_xMin, xMin, sizeof(f64) * Dim, cudaMemcpyHostToDevice));
        ERRCHECK(cudaMemcpy(d_xMax, xMax, sizeof(f64) * Dim, cudaMemcpyHostToDevice));
        if (d_binEdges) {
            ERRCHECK(cudaMemcpy(d_binEdges, binEdges, sizeof(f64) * arraySum(nBinsAxis, Dim), cudaMemcpyHostToDevice));
        }
        ERRCHECK(cudaMemcpy(d_binEdgesIdx, binEdgesIdx, sizeof(isize) * Dim, cudaMemcpyHostToDevice));
    }

    ~SingleHisto() {
        ERRCHECK(cudaFree(d_histogram));
        ERRCHECK(cudaFree(d_nBinsAxis));
        ERRCHECK(cudaFree(d_xMin));
        ERRCHECK(cudaFree(d_xMax));
        ERRCHECK(cudaFree(d_binEdges));
        ERRCHECK(cudaFree(d_binEdgesIdx));
        ERRCHECK(cudaFree(d_coords));
        ERRCHECK(cudaFree(d_weights));
    }

    SingleHisto(const SingleHisto &) = delete;
    SingleHisto &operator=(const SingleHisto &) = delete;

    void RetrieveResults(T *histogram, f64 *stats) {
        ERRCHECK(cudaMemcpy(histogram, d_histogram, sizeof(T) * nBins, cudaMemcpyDeviceToHost));
    }

    void Fill(usize n, const f64 *coords) {
        if (n > maxBulkSize) {
            Fill(maxBulkSize, coords);
            Fill(n - maxBulkSize, coords + maxBulkSize);
            return;
        }

        ERRCHECK(cudaMemcpy(d_coords, coords, sizeof(f64) * n, cudaMemcpyHostToDevice));

        usize numBlocks = n % BlockSize == 0 ? n / BlockSize : n / BlockSize + 1;
        HistogramGlobal<T, Dim, op><<<numBlocks, BlockSize>>>(
            d_histogram,
            d_binEdges, d_binEdgesIdx, d_nBinsAxis,
            d_xMin, d_xMax,
            d_coords, nullptr, bulkDim, n
        );
        ERRCHECK(cudaPeekAtLastError());
    }

    void Fill(usize n, const f64 *coords, const f64 *weights) {
        assert(n <= maxBulkSize);

        ERRCHECK(cudaMemcpy(d_coords, coords, sizeof(f64) * bulkDim * n, cudaMemcpyHostToDevice));
        ERRCHECK(cudaMemcpy(d_weights, weights, sizeof(f64) * n, cudaMemcpyHostToDevice));

        usize numBlocks = n % BlockSize == 0 ? n / BlockSize : n / BlockSize + 1;
        HistogramGlobal<T, Dim, op><<<numBlocks, BlockSize>>>(
            d_histogram,
            d_binEdges, d_binEdgesIdx, d_nBinsAxis,
            d_xMin, d_xMax,
            d_coords, d_weights, bulkDim, n
        );
        ERRCHECK(cudaPeekAtLastError());
    }
};

int main() {
    usize nBins = 12;
    f64 xMin =   0.0;
    f64 xMax = 100.0;
    isize binEdgesIdx = -1;
    auto myHisto = SingleHisto<f64, add2, 1>(&nBins, &xMin, &xMax, nullptr, &binEdgesIdx, 2 , 256);

    auto refHisto = TH1D(
        "", "",
        nBins - 2, xMin, xMax
    );

    // Generate some data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<f64> coord_dist(0.0, 50.0);
    std::uniform_real_distribution<f64> weight_dist(0.0, 1.0);

    f64 coords[256 * 2], weights[256];
    for (usize i = 0; i < 100000; i += 256) {
        usize n = std::min(100000 - i, 256UL);
        for (usize j = 0; j < n; ++j) {
            coords[j] = coord_dist(gen);
            coords[j + n] = weight_dist(gen);
            weights[j] = weight_dist(gen);
        }
        myHisto.Fill(n, coords, weights);

        for (usize j = 0; j < n; ++j) {
            coords[j] = coords[j] + coords[j + n];
        }
        refHisto.FillN(n, coords, weights);
    }

    f64 myResult[nBins];
    myHisto.RetrieveResults(myResult, nullptr);
    printArray(myResult, nBins);
    printArray(refHisto.GetArray(), nBins);

    return 0;
}
