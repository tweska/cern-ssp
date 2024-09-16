#ifndef GBHISTO_H
#define GBHISTO_H

#include <vector>

#include "types.h"

using namespace std;

/// @brief CUDA batch histogramming class
/// @tparam T Histograms data type
/// @tparam BlockSize Cuda block size to use in the kernels
template <typename T, usize BlockSize = 256>
class GbHisto {
protected:
    // Buffers (histograms)
    T     *d_histograms;         ///< Histograms buffer (index using histoOffset)
    f64   *d_binEdges;           ///< Bin edges for each axis (len: sum(nBinsAxis), index: binEdgesOffset)

    // Limits
    f64   *d_xMin;               ///< Lower limit per axis (len sum(nDims), index: axisOffset)
    f64   *d_xMax;               ///< Upper limit per axis (len sum(nDims), index: axisOffset)

    // Lengths
    usize    nHistos;            ///< Total number of histograms
    usize    nBins;              ///< Total number of bins
    usize    nAxis;              ///< Total number of axis
    usize *d_nDims;              ///< Number of dimensions for each histogram (len: nHistos)
    usize *d_nBinsAxis;          ///< Number of bins per axis for each histogram (len: sum(nDims), index: axisOffset)

    // Offsets
    usize *h_histoResultOffset;  ///< Start index of each histogram in histograms (len: nHistos)
    usize *d_histoResultOffset;  ///< Start index of each histogram in histograms (len: nHistos)
    usize *h_histoOffset;        ///< Start index of each axis in  (len: nHistos)
    usize *d_histoOffset;        ///< Start index of each axis in  (len: nHistos)
    isize *d_binEdgesOffset;     ///< Start index of each binEdges for each axis (len: sum(nDims), index: axisOffset)

    // Buffers (processing)
    f64   *d_coords;             ///< Coordinates in xxx,yyy,zzz format (len: maxBulkSize * sum(nDims))
    f64   *d_weights;            ///< Weights for each coordinate (len: maxBulkSize * nHistos)

    usize    maxBulkSize;        ///< Maximum size of bulk to process at once

    f64 runtimeInit     = 0.0;
    f64 runtimeTransfer = 0.0;
    f64 runtimeKernel   = 0.0;
    f64 runtimeResult   = 0.0;

public:
    GbHisto() = delete;
    GbHisto(
        usize nHistos, const usize *nDims, const usize *nBinsAxis,
        const f64 *xMin, const f64 *xMax,
        const f64 *binEdges, const isize *binEdgesOffset,
        usize maxBulkSize=256
    );
    GbHisto(
        usize nHistos, const vector<usize> &nDims, const vector<usize> &nBinsAxis,
        const vector<f64> &xMin, const vector<f64> &xMax,
        const vector<f64> &binEdges, const vector<isize> &binEdgesOffset,
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
    void FillN(usize n, const f64 *coords, const f64 *weights = nullptr);
};

#endif //GBHISTO_H
