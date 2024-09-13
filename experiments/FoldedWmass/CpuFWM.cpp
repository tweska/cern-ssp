#include <vector>
#include <thread>

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <TTreePerfStats.h>

#include "types.h"
#include "timer.h"

#include "coords.h"
#include "CpuFWM.h"

#define ISOLATION_CRITICAL 0.5
#define NBINS 100
#define XMIN 0
#define XMAX 400
#define THREADS 16
#define BATCH_SIZE (2048 * THREADS)
#define RUNS 1

void bulkThread(const DefCoords *defCoords, TH1D **histo, const usize n, const usize tid) {
    const usize start = BATCH_SIZE / THREADS * tid;
    const usize end = std::min(start + BATCH_SIZE / THREADS, n);

    for (usize k = start; k < end; ++k) {
        const DefCoords dc = defCoords[k];
        for (i32 s = 0; s < 100; ++s) {
            for (i32 r = 0; r < 100; ++r) {
                const f32 scale = 0.9 + s*0.01;
                const f32 resolution = 0.8 + r*0.02;

                const f32 mass = foldedMass(
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

void FoldedWmass(Timer<> *timer, b8 print = false)
{
    TChain* chainReco = new TChain("reco");
    TChain* chainTruth = new TChain("particleLevel");

    chainReco->AddFile("data/output.root");
    chainTruth->AddFile("data/output.root");
    chainTruth->BuildIndex("eventNumber");
    chainReco->AddFriend(chainTruth);

    auto df = ROOT::RDataFrame(*chainReco).Filter(
        "TtbarLjets_spanet_up_index_NOSYS >= 0 && TtbarLjets_spanet_down_index_NOSYS >= 0"
    );

    auto truePt = [](
        const std::vector<f32>& truePt,
        const i32 index,
        const std::vector<f32>& recoDeltaR,
        const std::vector<i32>& trueIndex,
        const std::vector<f32>& trueDeltaR
    ) {
        i32 trueI1 = trueIndex[index];
        if (   recoDeltaR[index] < ISOLATION_CRITICAL
            || trueI1 < 0
            || static_cast<u32>(trueI1) >= truePt.size())
        {
            return -1.0f;
        }

        f32 trueIsol1 = trueDeltaR[trueI1];
        if (   static_cast<u32>(trueI1) >= trueDeltaR.size()
            || trueIsol1 < ISOLATION_CRITICAL)
        {
            return -1.0f;
        }

        return truePt[trueI1];
    };

    df = df.Define(
        "truePt1",
        truePt,
        {
            "particleLevel.jet_pt",
            "TtbarLjets_spanet_up_index_NOSYS",
            "jet_reco_to_reco_jet_closest_dR_NOSYS",
            "jet_truth_jet_paired_index",
            "jet_truth_to_truth_jet_closest_dR"
        }
    ).Define(
        "truePt2",
        truePt,
        {
            "particleLevel.jet_pt",
            "TtbarLjets_spanet_down_index_NOSYS",
            "jet_reco_to_reco_jet_closest_dR_NOSYS",
            "jet_truth_jet_paired_index",
            "jet_truth_to_truth_jet_closest_dR"
        }
    );

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

    usize i = 0;
    DefCoords *defCoords = new DefCoords[BATCH_SIZE];
    df.Foreach(
        [&i, defCoords, histos, timer] (
        const std::vector<f32>& recoPt,
        const std::vector<f32>& recoEta,
        const std::vector<f32>& recoPhi,
        const std::vector<f32>& recoE,
        const i32 i1, const i32 i2,
        const f32 truePt1, const f32 truePt2
    ) {
        defCoords[i] = {
            recoPt[i1], recoEta[i1], recoPhi[i1], recoE[i1],
            recoPt[i2], recoEta[i2], recoPhi[i2], recoE[i2],
            truePt1, truePt2
        };

        if (++i == BATCH_SIZE) {
            timer->start();
            processBulk(defCoords, histos, BATCH_SIZE);
            timer->pause();
            i = 0;
        }
    },
    {
        "jet_pt_NOSYS", "jet_eta", "jet_phi", "jet_e_NOSYS",
        "TtbarLjets_spanet_up_index_NOSYS", "TtbarLjets_spanet_down_index_NOSYS",
        "truePt1", "truePt2"
    });

    // Process the last batch!
    if (i != 0) {
        if (timer) timer->start();
        processBulk(defCoords, histos, i);
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
    delete[] defCoords;
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

    if (warmupFlag || printFlag) {
        FoldedWmass(nullptr, printFlag);
    }

    Timer<> timer[RUNS];
    for (auto i = 0; i < RUNS; ++i) {
        FoldedWmass(&timer[i]);
    }
    std::cerr << "Define + Fill "; printTimerMinMaxAvg(timer, RUNS);

    return 0;
}
