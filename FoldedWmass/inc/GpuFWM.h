#ifndef GPUFWM_H
#define GPUFWM_H

#include "types.h"
#include "timer.h"

#include "coords.h"

template <usize BlockSize = 64, usize MaxBulkSize = 131072>
class GpuFWM {
protected:
    f64 *d_histos;
    usize nBins;

    f64 xMin;
    f64 xMax;

    f32 *d_scales;
    f32 *d_resolutions;
    usize nScales;
    usize nResolutions;

    DefCoords *d_defCoords;

    Timer<> *rtTransfer;
    Timer<> *rtKernel;
    Timer<> *rtResult;

public:
    GpuFWM() = delete;
    GpuFWM(
        usize nBins,
        f64 xMin, f64 xMax,
        f32 *scales, usize nScales,
        f32 *resolutions, usize nResolutions,
        Timer<> *rtSetup = nullptr,
        Timer<> *rtKernel = nullptr,
        Timer<> *rtResult = nullptr
    );
    ~GpuFWM();

    GpuFWM(const GpuFWM &) = delete;
    GpuFWM &operator=(const GpuFWM &) = delete;

    void RetrieveResult(usize i, f64 *histogram);
    void RetrieveResults(f64 *histogram);
    void FillN(usize n, const DefCoords *defCoords);
};

#endif //GPUFWM_H
