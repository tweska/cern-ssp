#ifndef GHISTO_H
#define GHISTO_H

#include "../../inc/types.h"

typedef f64 (Op)(f64*, usize);
f64 InvariantCoordMasses(f64 *coords, size_t n);

template <typename T, Op op, usize BlockSize = 256>
class GHisto {
protected:
    T     *d_histogram;   ///< Histogram buffer
    usize    nBins;       ///< Total number of bins in the histogram (with under/overflow)

    f64      xMin;        ///< Lower edge of first bin
    f64      xMax;        ///< Upper edge of last bin

    f64   *d_coords;      ///< Coordinates buffer
    f64   *d_weights;     ///< Weights buffer

    usize    bulkDims;    ///< Number of dimensions in the coordinate buffer
    usize    maxBulkSize; ///< Size of the coordinates buffer

public:
    GHisto() = delete;
    GHisto(
        usize nBins,
        f64 xMin, f64 xMax,
        usize bulkDims, usize maxBulkSize = 256
    );
    ~GHisto();

    GHisto(const GHisto &) = delete;
    GHisto &operator=(const GHisto &) = delete;

    void RetrieveResults(T *histogram);
    void FillN(usize n, const f64 *coords, const f64 *weights = nullptr);
};

using GHistoIM = GHisto<double, InvariantCoordMasses>;

#endif //GHISTO_H
