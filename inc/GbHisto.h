#ifndef GBHISTO_H
#define GBHISTO_H

#include <vector>

#include "types.h"

using namespace std;

/// @brief CUDA batch histogramming class
/// @tparam T Histograms data type
/// @tparam BlockSize Cuda block size to use in the kernels
template <typename T, u32 BlockSize = 256>
class GbHisto {
protected:
    // Buffers (histograms)
    T   *d_histograms;         ///< Histograms buffer (index using histoOffset)
    f64 *d_binEdges;           ///< Bin edges for each axis (len: sum(nBinsAxis), index: binEdgesOffset)

    // Limits
    f64 *d_xMin;               ///< Lower limit per axis (len sum(nDims), index: axisOffset)
    f64 *d_xMax;               ///< Upper limit per axis (len sum(nDims), index: axisOffset)

    // Lengths
    u32    nHistos;            ///< Total number of histograms
    u32    nBins;              ///< Total number of bins
    u32    nAxis;              ///< Total number of axis
    u32 *d_nDims;              ///< Number of dimensions for each histogram (len: nHistos)
    u32 *d_nBinsAxis;          ///< Number of bins per axis for each histogram (len: sum(nDims), index: axisOffset)

    // Offsets
    u32 *h_histoResultOffset;  ///< Start index of each histogram in histograms (len: nHistos)
    u32 *d_histoResultOffset;  ///< Start index of each histogram in histograms (len: nHistos)
    u32 *h_histoOffset;        ///< Start index of each axis in  (len: nHistos)
    u32 *d_histoOffset;        ///< Start index of each axis in  (len: nHistos)
    i32 *d_binEdgesOffset;     ///< Start index of each binEdges for each axis (len: sum(nDims), index: axisOffset)

    // Buffers (processing)
    f64 *d_coords;             ///< Coordinates in xxx,yyy,zzz format (len: maxBulkSize * sum(nDims))
    f64 *d_weights;            ///< Weights for each coordinate (len: maxBulkSize)  TODO: Should it be maxBulkSize * nHistos?

    usize  maxBulkSize;        ///< Maximum size of bulk to process at once

    f64 runtimeInit     = 0.0;
    f64 runtimeTransfer = 0.0;
    f64 runtimeKernel   = 0.0;
    f64 runtimeResult   = 0.0;

public:
    GbHisto() = delete;
    GbHisto(
        u32 nHistos, const u32 *nDims, const u32 *nBinsAxis,
        const f64 *xMin, const f64 *xMax,
        const f64 *binEdges, const i32 *binEdgesOffset,
        usize maxBulkSize=256
    );
    GbHisto(
        u32 nHistos, const vector<u32> &nDims, const vector<u32> &nBinsAxis,
        const vector<f64> &xMin, const vector<f64> &xMax,
        const vector<f64> &binEdges, const vector<i32> &binEdgesOffset,
        usize maxBulkSize=256
    ) {
        GbHisto(
            nHistos, nDims.data(), nBinsAxis.data(),
            xMin.data(), xMax.data(),
            binEdges.data(), binEdgesOffset.data(),
            maxBulkSize
        );
    }
    ~GbHisto();

    GbHisto(const GbHisto &) = delete;
    GbHisto &operator=(const GbHisto &) = delete;

    void RetrieveResults(T *histogram, f64 *stats = nullptr);
    void GetRuntime(f64 *runtimeInit, f64 *runtimeTransfer, f64 *runtimeKernel, f64 *runtimeResult);
    void PrintRuntime(std::ostream &output = std::cout);
    void Fill(u32 n, const f64 *coords);
    void Fill(u32 n, const f64 *coords, const f64 *weights);
};

#endif //GBHISTO_H
