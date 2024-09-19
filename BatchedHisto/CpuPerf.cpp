#include <random>

// ROOT
#include "TH1D.h"
#include "TH2D.h"

#include "types.h"
#include "timer.h"

#define RUNS 10
#define BULKSIZE 32768
#define N 100000000

void CpuPerf(Timer<> *timer) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> coords_dis(-10.0, 110.0);
    std::uniform_real_distribution<> weight_dis(0.0, 1.0);

    const usize nBinsAxis[] = {100, 50, 50, 50, 50};
    const f64 xMin[] = {0, 0, 0, 0, 0};
    const f64 xMax[] = {100, 100, 100, 100, 100};

    auto histo1 = TH1D(
        "", "",
        nBinsAxis[0] - 2, xMin[0], xMax[0]
    );
    auto histo2 = TH2D(
        "", "",
        nBinsAxis[1] - 2, xMin[1], xMax[1],
        nBinsAxis[2] - 2, xMin[2], xMax[2]
    );
    auto histo3 = TH2D(
        "", "",
        nBinsAxis[3] - 2, xMin[3], xMax[3],
        nBinsAxis[4] - 2, xMin[4], xMax[4]
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
            if (timer) timer->start();
            histo1.FillN(BULKSIZE, &coords[0 * BULKSIZE], weights);
            histo2.FillN(BULKSIZE, &coords[1 * BULKSIZE], &coords[2 * BULKSIZE], weights);
            histo2.FillN(BULKSIZE, &coords[3 * BULKSIZE], &coords[4 * BULKSIZE], weights);
            if (timer) timer->pause();
            i = 0;
        }
    }

    if (i != 0) {
        const usize n = i / 5;
        if (timer) timer->start();
        histo1.FillN(n, &coords[0 * n], weights);
        histo2.FillN(n, &coords[1 * n], &coords[2 * n], weights);
        histo2.FillN(n, &coords[3 * n], &coords[4 * n], weights);
        if (timer) timer->pause();
    }

    delete[] coords;
    delete[] weights;
}

int main() {
    Timer<> timer[RUNS];

    // Warmup...
    CpuPerf(nullptr);
    for (usize i = 0; i < RUNS; ++i) {
        CpuPerf(&timer[i]);
    }

    // Print timing results
    std::cout << "Fill     "; printTimerMinMaxAvg(timer, RUNS);
}
