#include <random>

#include "types.h"
#include "timer.h"

#include "GbHisto.h"

#define RUNS 10
#define BULKSIZE 32768
#define N 100000000

void GpuPerf(Timer<> *rtInit, Timer<> *rtTransfer, Timer<> *rtKernel, Timer<> *rtResult) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> coords_dis(-10.0, 110.0);
    std::uniform_real_distribution<> weight_dis(0.0, 1.0);

    const usize nDims[] = {1, 2, 2};
    const usize nBinsAxis[] = {100, 50, 50, 50, 50};
    const f64 xMin[] = {0, 0, 0, 0, 0};
    const f64 xMax[] = {100, 100, 100, 100, 100};
    const isize binEdgesOffset[] = {-1, -1, -1, -1, -1};

    auto gpuHisto = new GbHisto<f64>(
        3, nDims, nBinsAxis,
        xMin, xMax,
        nullptr, binEdgesOffset,
        32768,
        rtInit, rtTransfer, rtKernel, rtResult
    );

    auto *coords = new f64[BULKSIZE * 5];
    auto *weights = new f64[BULKSIZE];
    usize i = 0;
    for (usize n = 0; n < N; ++n) {
        coords[i++] = coords_dis(gen);
        coords[i++] = coords_dis(gen);
        coords[i++] = coords_dis(gen);
        coords[i++] = coords_dis(gen);
        coords[i++] = coords_dis(gen);
        weights[n % BULKSIZE] = weight_dis(gen);

        if (i == BULKSIZE * 5) {
            gpuHisto->FillN(BULKSIZE, coords, weights);
            i = 0;
        }
    }

    if (i != 0) {
        gpuHisto->FillN(i / 5, coords, weights);
    }

    delete gpuHisto;
    delete[] coords;
    delete[] weights;
}

int main() {
    Timer<> rtsInit[RUNS], rtsTransfer[RUNS], rtsKernel[RUNS], rtsResult[RUNS];

    // Warmup...
    GpuPerf(nullptr, nullptr, nullptr, nullptr);
    for (usize i = 0; i < RUNS; ++i) {
        GpuPerf(&rtsInit[i], &rtsTransfer[i], &rtsKernel[i], &rtsResult[i]);
    }

    // Print timing results
    std::cout << "Init     "; printTimerMinMaxAvg(rtsInit, RUNS);
    std::cout << "Transfer "; printTimerMinMaxAvg(rtsTransfer, RUNS);
    std::cout << "Fill     "; printTimerMinMaxAvg(rtsKernel, RUNS);
    std::cout << "Result   "; printTimerMinMaxAvg(rtsResult, RUNS);

    Timer<> rtsTotal[RUNS];
    for (usize i = 0; i < RUNS; ++i) {
        rtsTotal[i] = rtsInit[i] + rtsTransfer[i] + rtsKernel[i] + rtsResult[i];
    }
    std::cerr << "Total    "; printTimerMinMaxAvg(rtsTotal, RUNS);

    return 0;
}
