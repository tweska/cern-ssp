#ifndef GHISTO_H
#define GHISTO_H

#include "types.h"

/// @brief CUDA histogramming class
/// @tparam T Histogram data type
/// @tparam Dim Dimensionality of the histogram
/// @tparam BlockSize Cuda block size to use in the kernels
template <typename T, u32 Dim, u32 BlockSize = 256>
class GHisto {
private:
    T   *d_histogram;   ///< Histogram buffer
    u32    nBins;       ///< Total number of bins in the histogram (with under/overflow)

    u32 *d_nBinsAxis;   ///< Number of bins per axis (with under/overflow)
    f64 *d_xMin;        ///< Lower edge of first bin per axis
    f64 *d_xMax;        ///< Upper edge of last bin per axis
    f64 *d_binEdges;    ///< Bin edges array for each axis (may be null for fixed bins)
    i32 *d_binEdgesIdx; ///< Start index of the binedges in binEdges (-1 for fixed bins)

    f64 *d_coords;      ///< Coordinates buffer
    f64 *d_weights;     ///< Weights buffer

    usize maxBulkSize;  ///< Size of the coordinates buffer

public:
    GHisto() = delete;
    GHisto(
        const u32 *nBinsAxis,
        const f64 *xMin, const f64 *xMax,
        const f64 *binEdges, const i32 *binEdgesIdx,
        usize maxBulkSize=256
    );
    ~GHisto();

    GHisto(const GHisto &) = delete;
    GHisto &operator=(const GHisto &) = delete;

    void RetrieveResults(T *histogram, f64 *stats);
    void Fill(u32 n, const f64 *coords);
    void Fill(u32 n, const f64 *coords, const f64 *weights);
};

#endif //GHISTO_H
