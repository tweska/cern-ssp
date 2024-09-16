#include <cstring>
#include <vector>
#include <iostream>
#include <iomanip>

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDFHelpers.hxx>

#include "types.h"
#include "timer.h"
#include "util.h"

#include "coords.h"
#include "GpuFWM.h"

#define NBINS 100
#define XMIN 0
#define XMAX 400
#define INPUT_SIZE 100000
#define BATCH_SIZE 32768
#define RUNS 10

void FoldedWmass(DefCoords *defCoords, Timer<> *rtTransfer, Timer<> *rtKernel, Timer<> *rtResult, b8 print = false)
{
    f32 scales[100], resolutions[100];
    for (i32 s = 0; s < 100; ++s) { scales[s] = 0.9f + static_cast<f32>(s) * 0.01f; }
    for (i32 r = 0; r < 100; ++r) { resolutions[r] = 0.8f + static_cast<f32>(r) * 0.02f; }
    auto gpuFWM = GpuFWM(
        NBINS, XMIN, XMAX,
        scales, 100, resolutions, 100,
        rtTransfer, rtKernel, rtResult
    );

    // Process the batches.
    for (usize i = 0; i < INPUT_SIZE; i += BATCH_SIZE) {
        const usize n = std::min<usize>(BATCH_SIZE, INPUT_SIZE - i);
        gpuFWM.FillN(n, &defCoords[i]);
    }

    // Retrieve the results.
    f64 *histoValues = new f64[10000 * (NBINS + 2)];
    gpuFWM.RetrieveResults(histoValues);

    if (print) {
        for (int s = 0; s < 100; ++s) {
            for (int r = 0; r < 100; ++r) {
                std::cout << "r=" << std::setw(2) << r << " s=" << std::setw(2) << s << " : ";
                printArray(&histoValues[(r * 100 + s) * (NBINS + 2)], NBINS + 2);
            }
        }
    }

    delete[] histoValues;
}

i32 main(i32 argc, c8 *argv[])
{
    b8 warmupFlag = false;
    b8 printFlag = false;
    usize runs = RUNS;
    for (i32 i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--warmup") == 0) { warmupFlag = true; }
        else if (strcmp(argv[i], "--print") == 0) { printFlag = true; }
        else { runs = atol(argv[i]); }
    }

    DefCoords *defCoords = new DefCoords[INPUT_SIZE];
    getCoords(defCoords, INPUT_SIZE);

    if (warmupFlag || printFlag) {
        FoldedWmass(defCoords, nullptr, nullptr, nullptr, printFlag);
    }

    if (runs < 1) {
        delete[] defCoords;
        return 0;
    }

    Timer<> rtsTransfer[runs], rtsKernel[runs], rtsResult[runs];
    for (usize i = 0; i < runs; ++i) {
        FoldedWmass(defCoords, &rtsTransfer[i], &rtsKernel[i], &rtsResult[i]);
    }
    std::cerr << "Transfer      "; printTimerMinMaxAvg(rtsTransfer, runs);
    std::cerr << "Define + Fill "; printTimerMinMaxAvg(rtsKernel, runs);
    std::cerr << "Result        "; printTimerMinMaxAvg(rtsResult, runs);

    Timer<> rtsTotal[runs];
    for (usize i = 0; i < runs; ++i) {
        rtsTotal[i] = rtsTransfer[i] + rtsKernel[i] + rtsResult[i];
    }
    std::cerr << "Total         "; printTimerMinMaxAvg(rtsTotal, runs);

    delete[] defCoords;
    return 0;
}
