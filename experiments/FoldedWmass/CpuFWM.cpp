#include <vector>
#include <thread>

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDFHelpers.hxx>

#include "types.h"
#include "timer.h"

#include "coords.h"
#include "CpuFWM.h"

#define NBINS 100
#define XMIN 0
#define XMAX 400
#define THREADS 16
#define INPUT_SIZE 2062
#define BATCH_SIZE (128 * THREADS)
#define RUNS 1

void bulkThread(const DefCoords *defCoords, TH1D **histo, const usize n, const usize tid) {
    const usize start = BATCH_SIZE / THREADS * tid;
    const usize end = std::min(start + BATCH_SIZE / THREADS, n);

    for (usize k = start; k < end; ++k) {
        const DefCoords dc = defCoords[k];
        for (i32 s = 0; s < 100; ++s) {
            for (i32 r = 0; r < 100; ++r) {
                const f32 scale = 0.9f + static_cast<f32>(s) * 0.01f;
                const f32 resolution = 0.8f + static_cast<f32>(r) * 0.02f;

                const f64 mass = foldedMass(
                    dc.recoPt1, dc.recoEta1, dc.recoPhi1, dc.recoE1,
                    dc.recoPt2, dc.recoEta2, dc.recoPhi2, dc.recoE2,
                    dc.truePt1, dc.truePt2,
                    scale, resolution
                );
                histo[s * 100 + r]->Fill(mass);
            }
        }
    }
}

void processBulk(DefCoords *defCoords, std::vector<TH1D*> histos, usize n) {
    std::vector<std::thread> threads;
    threads.reserve(THREADS);
    for (usize tid = 0; tid < THREADS; ++tid) {
        threads.emplace_back(bulkThread, defCoords, &histos[tid * 10000], n, tid);
    }

    for (auto &thread : threads) {
        thread.join();
    }
}

void FoldedWmass(DefCoords *defCoords, Timer<> *timer, b8 print = false)
{
    std::vector<TH1D*> histos;
    histos.reserve(10000 * THREADS);
    for (usize tid = 0; tid < THREADS; ++tid) {
        for (i32 s = 0; s < 100; ++s) {
            for (i32 r = 0; r < 100; ++r) {
                histos.push_back(new TH1D(
                    ("FWM_" + std::to_string(tid) + "_" + std::to_string(s) + "_" + std::to_string(r)).c_str(),
                    "FWM;m;N_{Events}",
                    NBINS, XMIN, XMAX
                ));
            }
        }
    }

    // Process the batches.
    for (usize i = 0; i < INPUT_SIZE; i += BATCH_SIZE) {
        const usize n = std::min<usize>(BATCH_SIZE, INPUT_SIZE - i);
        if (timer) timer->start();
        processBulk(&defCoords[i], histos, n);
        if (timer) timer->pause();
    }

    std::vector<TH1D*> mergedHistos;
    mergedHistos.reserve(10000);
    if (timer) timer->start();
    for (i32 s = 0; s < 100; ++s) {
        for (i32 r = 0; r < 100; ++r) {
            TH1D *mergedHisto = new TH1D(
                ("Merged_FWM_" + std::to_string(s) + "_" + std::to_string(r)).c_str(),
                "FWM;m;N_{Events}",
                NBINS, XMIN, XMAX
            );
            mergedHistos.emplace_back(mergedHisto);
            for (usize tid = 0; tid < THREADS; ++tid) {
                mergedHisto->Add(histos[(10000 * tid) + s * 100 + r]);
            }
        }
    }

    // Retrieve the results.
    f64 **results = new f64*[10000];
    for (usize i = 0; i < 10000; ++i) {
        results[i] = mergedHistos[i]->GetArray();
    }
    if (timer) timer->pause();

    if (print) {
        for (int s = 0; s < 100; ++s) {
            for (int r = 0; r < 100; ++r) {
                std::cout << "r=" << std::setw(2) << r << " s=" << std::setw(2) << s << " : ";
                printArray(results[r * 100 + s], NBINS + 2);
            }
        }
    }

    delete[] results;
}

i32 main(i32 argc, c8 *argv[])
{
    TH1::AddDirectory(false);

    b8 warmupFlag = false;
    b8 printFlag = false;
    for (i32 i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--warmup") == 0) { warmupFlag = true; }
        if (strcmp(argv[i], "--print") == 0) { printFlag = true; }
    }

    DefCoords *defCoords = new DefCoords[INPUT_SIZE];
    getCoords(defCoords, INPUT_SIZE);

    if (warmupFlag || printFlag) {
        FoldedWmass(defCoords, nullptr, printFlag);
    }

    Timer<> timer[RUNS];
    for (auto i = 0; i < RUNS; ++i) {
        FoldedWmass(defCoords, &timer[i]);
    }
    std::cerr << "Define + Fill "; printTimerMinMaxAvg(timer, RUNS);

    delete[] defCoords;
    return 0;
}
