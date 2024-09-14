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

#define ISOLATION_CRITICAL 0.5
#define MAX_BULK_SIZE 32768
#define NBINS 100
#define XMIN 0
#define XMAX 400
#define RUNS 10

void FoldedWmass(Timer<> *rtTransfer, Timer<> *rtKernel, Timer<> *rtResult, b8 print = false)
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

    f32 scales[100], resolutions[100];
    for (i32 s = 0; s < 100; ++s) { scales[s] = 0.9f + static_cast<f32>(s) * 0.01f; }
    for (i32 r = 0; r < 100; ++r) { resolutions[r] = 0.8f + static_cast<f32>(r) * 0.02f; }
    auto gpuFWM = GpuFWM(
        NBINS, XMIN, XMAX,
        scales, 100, resolutions, 100,
        rtTransfer, rtKernel, rtResult
    );

    usize i = 0;
    DefCoords *defCoords = new DefCoords[MAX_BULK_SIZE];
    df.Foreach(
        [&i, defCoords, &gpuFWM] (
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

        if (++i == MAX_BULK_SIZE) {
            gpuFWM.FillN(MAX_BULK_SIZE, defCoords);
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
        gpuFWM.FillN(i, defCoords);
    }

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
    delete[] defCoords;
}

i32 main(i32 argc, c8 *argv[])
{
    b8 warmupFlag = false;
    b8 printFlag = false;
    for (i32 i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--warmup") == 0) { warmupFlag = true; }
        if (strcmp(argv[i], "--print") == 0) { printFlag = true; }
    }

    if (warmupFlag || printFlag) {
        FoldedWmass(nullptr, nullptr, nullptr, printFlag);
    }

    Timer<> rtsTransfer[RUNS], rtsKernel[RUNS], rtsResult[RUNS];
    for (auto i = 0; i < RUNS; ++i) {
        FoldedWmass(&rtsTransfer[i], &rtsKernel[i], &rtsResult[i]);
    }
    std::cerr << "Transfer      "; printTimerMinMaxAvg(rtsTransfer, RUNS);
    std::cerr << "Define + Fill "; printTimerMinMaxAvg(rtsKernel, RUNS);
    std::cerr << "Result        "; printTimerMinMaxAvg(rtsResult, RUNS);

    Timer<> rtsTotal[RUNS];
    for (auto i = 0; i < RUNS; ++i) {
        rtsTotal[i] = rtsTransfer[i] + rtsKernel[i] + rtsResult[i];
    }
    std::cerr << "Total         "; printTimerMinMaxAvg(rtsTotal, RUNS);

    return 0;
}
