#ifndef GPUFWM_H
#define GPUFWM_H

#include "types.h"
#include "timer.h"

typedef struct
{
    f32 recoPt1; f32 recoEta1; f32 recoPhi1; f32 recoE1;
    f32 recoPt2; f32 recoEta2; f32 recoPhi2; f32 recoE2;
    f32 truePt1; f32 truePt2;  // May be negative to indicate invalid values!
} DefCoords;

template <usize BlockSize = 256, usize MaxBulkSize = 32768>
class GpuFWM {
protected:
    f64 *d_histos;
    usize nBins;

    f32 xMin;
    f32 xMax;

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
        f32 xMin, f32 xMax,
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
