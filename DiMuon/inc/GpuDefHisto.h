#ifndef GPUDEFHISTO_H
#define GPUDEFHISTO_H

#include "types.h"
#include "timer.h"

typedef f64 (Op)(f64*, usize, usize);

template <typename T, Op op, usize BlockSize = 256>
class GpuDefHisto {
protected:
    T     *d_histogram;   ///< Histogram buffer
    usize    nBins;       ///< Total number of bins in the histogram (with under/overflow)

    f64      xMin;        ///< Lower edge of first bin
    f64      xMax;        ///< Upper edge of last bin

    f64   *d_coords;      ///< Coordinates buffer
    f64   *d_weights;     ///< Weights buffer

    usize    bulkDims;    ///< Number of dimensions in the coordinate buffer
    usize    maxBulkSize; ///< Size of the coordinates buffer

    Timer<> *timerTransfer;
    Timer<> *timerFill;
    Timer<> *timerResult;

public:
    GpuDefHisto() = delete;
    GpuDefHisto(
        usize nBins,
        f64 xMin, f64 xMax,
        usize bulkDims, usize maxBulkSize,
        Timer<> *timerTransfer = nullptr,
        Timer<> *timerFill = nullptr,
        Timer<> *timerResult = nullptr
    );
    ~GpuDefHisto();

    GpuDefHisto(const GpuDefHisto &) = delete;
    GpuDefHisto &operator=(const GpuDefHisto &) = delete;

    void RetrieveResults(T *histogram);
    usize ExecuteOp(const f64 *coord);
    void FillN(usize n, const f64 *coords, const f64 *weights = nullptr);
};

#endif //GPUDEFHISTO_H
