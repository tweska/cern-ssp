#include "CUDAHelpers.cuh"

#include "GpuFWM.h"

#define ISOLATION_CRITICAL 0.5

__device__
inline f32 angle(
    const f32 x1, const f32 y1, const f32 z1,
    const f32 x2, const f32 y2, const f32 z2
) {
    // cross product
    const f32 cx = y1 * z2 - y2 * z1;
    const f32 cy = x1 * z2 - x2 * z1;
    const f32 cz = x1 * y2 - x2 * y1;

    return atan2(
        sqrt(cx * cx + cy * cy + cz * cz),  // norm of cross product
        x1 * x2 + y1 * y2 + z1 * z2         // dot product
    );
}

__device__
inline f32 invariantMassPxPyPzM(
   const f32 x1, const f32 y1, const f32 z1, const f32 mass1,
   const f32 x2, const f32 y2, const f32 z2, const f32 mass2
) {
    // Numerically stable computation of Invariant Masses
    const f32 pp1 = x1 * x1 + y1 * y1 + z1 * z1;
    const f32 pp2 = x2 * x2 + y2 * y2 + z2 * z2;

    if (pp1 <= 0 && pp2 <= 0)
        return (mass1 + mass2);
    if (pp1 <= 0) {
        f32 mm = mass1 + sqrt(mass2*mass2 + pp2);
        f32 m2 = mm*mm - pp2;
        return m2 >= 0 ? sqrt(m2) : sqrt(-m2);
    }
    if (pp2 <= 0) {
        f32 mm = mass2 + sqrt(mass1*mass1 + pp1);
        f32 m2 = mm*mm - pp1;
        return m2 >= 0 ? sqrt(m2) : sqrt(-m2);
    }

    const f32 mm1 = mass1 * mass1;
    const f32 mm2 = mass2 * mass2;

    const f32 r1 = mm1 / pp1;
    const f32 r2 = mm2 / pp2;
    const f32 x = r1 + r2 + r1 * r2;
    const f32 a = angle(x1, y1, z1, x2, y2, z2);
    const f32 cos_a = cos(a);
    f32 y;
    if (cos_a >= 0){
        y = (x + sin(a) * sin(a)) / (sqrt(x + 1) + cos_a);
    } else {
        y = sqrt(x + 1) - cos_a;
    }

    const f32 z = 2.0f * sqrt(pp1 * pp2);

    // Return invariant mass with (+, -, -, -) metric
    return sqrt(mm1 + mm2 + y * z);
}

__device__
inline f32 invariantMassPxPyPzE(
   const f32 x1, const f32 y1, const f32 z1, const f32 e1,
   const f32 x2, const f32 y2, const f32 z2, const f32 e2
) {
    f32 mass1, mass2;
    {
        const f32 pp1 = x1 * x1 + y1 * y1 + z1 * z1;
        const f32 mm1 = e1 * e1 - pp1;
        mass1 = (mm1 >= 0) ? sqrt(mm1) : 0;
    }
    {
        const f32 pp2 = x2 * x2 + y2 * y2 + z2 * z2;
        const f32 mm2 = e2 * e2 - pp2;
        mass2 = (mm2 >= 0) ? sqrt(mm2) : 0;
    }

    return invariantMassPxPyPzM(x1, y1, z1, mass1, x2, y2, z2, mass2);
}

#ifdef UNSTABLE_INVARIANT_MASS
__device__
f32 foldedMass(
    f32 recoPt1, const f32 recoEta1, const f32 recoPhi1, const f32 recoE1,
    f32 recoPt2, const f32 recoEta2, const f32 recoPhi2, const f32 recoE2,
    const f32 truePt1, const f32 truePt2,
    const f32 scale, const f32 resolution
) {
    // Apply forward folding if both truePt values are valid.
    if (truePt1 >= 0 && truePt2 >= 0) {
        recoPt1 = scale * recoPt1 + (recoPt1 - truePt1) * (resolution - scale);
        recoPt2 = scale * recoPt2 + (recoPt2 - truePt2) * (resolution - scale);
    }

    // Compute and return the invariant mass.
    const f32 xSum = recoPt1 * cos(recoPhi1) + recoPt2 * cos(recoPhi2);
    const f32 ySum = recoPt1 * sin(recoPhi1) + recoPt2 * sin(recoPhi2);
    const f32 zSum = recoPt1 * sinh(recoEta1) + recoPt2 * sinh(recoEta2);
    const f32 eSum = recoE1 + recoE2;
    return sqrt(eSum * eSum - xSum * xSum - ySum * ySum - zSum * zSum) / 1e3f;
}
#else
__device__
f32 foldedMass(
    f32 recoPt1, const f32 recoEta1, const f32 recoPhi1, const f32 recoE1,
    f32 recoPt2, const f32 recoEta2, const f32 recoPhi2, const f32 recoE2,
    const f32 truePt1, const f32 truePt2,
    const f32 scale, const f32 resolution
) {
    // Apply forward folding if both truePt values are valid.
    if (truePt1 >= 0 && truePt2 >= 0) {
        recoPt1 = scale * recoPt1 + (recoPt1 - truePt1) * (resolution - scale);
        recoPt2 = scale * recoPt2 + (recoPt2 - truePt2) * (resolution - scale);
    }

    // Compute and return the invariant mass.
    return invariantMassPxPyPzE(
        recoPt1 * cos(recoPhi1),
        recoPt1 * sin(recoPhi1),
        recoPt1 * sinh(recoEta1),
        recoE1,
        recoPt2 * cos(recoPhi2),
        recoPt2 * sin(recoPhi2),
        recoPt2 * sinh(recoEta2),
        recoE2
    ) / 1e3f;
}
#endif

__global__
void FillKernel(
    f64 *histos, const usize nBins,
    const f64 xMin, const f64 xMax,
    const f32 *scales, const usize nScales,
    const f32 *resolutions, const usize nResolutions,
    const DefCoords *defCoords, const usize bulkSize
) {
    const usize tid = threadIdx.x + blockDim.x * blockIdx.x;
    const usize stride = blockDim.x * gridDim.x;

    for (usize k = tid; k < bulkSize; k += stride) {
        const DefCoords& dc = defCoords[k];
        for (usize i = 0; i < nScales; ++i) {
            for (usize j = 0; j < nResolutions; ++j) {
                // Calculate the invariant mass after folding.
                const f64 x = foldedMass(
                    dc.recoPt1, dc.recoEta1, dc.recoPhi1, dc.recoE1,
                    dc.recoPt2, dc.recoEta2, dc.recoPhi2, dc.recoE2,
                    dc.truePt1, dc.truePt2,
                    scales[i], resolutions[j]
                );

                // Fill the right bin.
                usize bin = 1 + static_cast<usize>((nBins - 2) * (x - xMin) / (xMax - xMin));
                if (x < xMin) bin = 0;
                if (!(x < xMax)) bin = nBins - 1;
                atomicAdd(&histos[(i * nScales + j) * nBins + bin], 1.0f);
            }
        }
    }
}

template <usize BlockSize, usize MaxBulkSize>
GpuFWM<BlockSize, MaxBulkSize>::GpuFWM(
    usize nBins,
    f64 xMin, f64 xMax,
    f32 *scales, usize nScales,
    f32 *resolutions, usize nResolutions,
    Timer<> *rtTransfer, Timer<> *rtKernel, Timer<> *rtResult
) {
    this->nBins = nBins + 2;
    this->xMin = xMin;
    this->xMax = xMax;
    this->nScales = nScales;
    this->nResolutions = nResolutions;

    ERRCHECK(cudaMalloc(&d_histos, sizeof(f64) * nScales * nResolutions * this->nBins));
    ERRCHECK(cudaMalloc(&d_scales, sizeof(f32) * nScales));
    ERRCHECK(cudaMalloc(&d_resolutions, sizeof(f32) * nResolutions));
    ERRCHECK(cudaMalloc(&d_defCoords, sizeof(DefCoords) * MaxBulkSize));

    ERRCHECK(cudaMemcpy(d_scales, scales, sizeof(f32) * nScales, cudaMemcpyHostToDevice));
    ERRCHECK(cudaMemcpy(d_resolutions, resolutions, sizeof(f32) * nResolutions, cudaMemcpyHostToDevice));
    ERRCHECK(cudaDeviceSynchronize());

    this->rtTransfer = rtTransfer;
    this->rtKernel = rtKernel;
    this->rtResult = rtResult;
}

template <usize BlockSize, usize MaxBulkSize>
GpuFWM<BlockSize, MaxBulkSize>::~GpuFWM()
{
    ERRCHECK(cudaFree(d_histos));
    ERRCHECK(cudaFree(d_scales));
    ERRCHECK(cudaFree(d_resolutions));
    ERRCHECK(cudaFree(d_defCoords));
}

template <usize BlockSize, usize MaxBulkSize>
void GpuFWM<BlockSize, MaxBulkSize>::RetrieveResult(const usize i, f64 *histograms)
{
    if (rtResult) rtResult->start();
    ERRCHECK(cudaMemcpy(histograms, d_histos + i * nBins, sizeof(f64) * nBins, cudaMemcpyDeviceToHost));
    ERRCHECK(cudaDeviceSynchronize());
    if (rtResult) rtResult->pause();
}

template <usize BlockSize, usize MaxBulkSize>
void GpuFWM<BlockSize, MaxBulkSize>::RetrieveResults(f64 *histograms)
{
    if (rtResult) rtResult->start();
    ERRCHECK(cudaMemcpy(histograms, d_histos, sizeof(f64) * nScales * nResolutions * nBins, cudaMemcpyDeviceToHost));
    ERRCHECK(cudaDeviceSynchronize());
    if (rtResult) rtResult->pause();
}

template <usize BlockSize, usize MaxBulkSize>
void GpuFWM<BlockSize, MaxBulkSize>::FillN(const usize n, const DefCoords *defCoords)
{
    assert(n <= MaxBulkSize);

    if (rtTransfer) rtTransfer->start();
    ERRCHECK(cudaMemcpy(d_defCoords, defCoords, sizeof(DefCoords) * n, cudaMemcpyHostToDevice));
    ERRCHECK(cudaDeviceSynchronize());
    if (rtTransfer) rtTransfer->pause();

    if (rtKernel) rtKernel->start();
    const usize nBlocks = n % BlockSize == 0 ? n / BlockSize : n / BlockSize + 1;
    FillKernel<<<nBlocks, BlockSize>>>(
        d_histos, nBins,
        xMin, xMax,
        d_scales, nScales,
        d_resolutions, nResolutions,
        d_defCoords, n
    );
    ERRCHECK(cudaDeviceSynchronize());
    if (rtKernel) rtKernel->pause();
    ERRCHECK(cudaPeekAtLastError());
}

template class GpuFWM<>;
