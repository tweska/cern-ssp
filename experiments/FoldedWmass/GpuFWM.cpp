#include <vector>

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <TTreePerfStats.h>

#include "types.h"

#include "GpuFWM.h"

#define ISOLATION_CRITICAL 0.5
#define MAX_BULK_SIZE 32768
#define NBINS 100
#define XMIN 0
#define XMAX 400

void FoldedWmass()
{
    TChain* chainReco = new TChain("reco");
    TChain* chainTruth = new TChain("particleLevel");

    chainReco->AddFile("data/output.root");
    chainTruth->AddFile("data/output.root");
    chainTruth->BuildIndex("eventNumber");
    chainReco->AddFriend(chainTruth);
    auto treeStats = new TTreePerfStats("ioperf", chainReco);

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
    for (i32 i = 0; i < 100; ++i) { scales[i] = 0.9 + i * 0.01; }
    for (i32 i = 0; i < 100; ++i) { resolutions[i] = 0.8 + i * 0.02; }
    auto gpuFWM = GpuFWM<256, MAX_BULK_SIZE>(NBINS, XMIN, XMAX, scales, 100, resolutions, 100);

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

    treeStats->Print();

    delete[] histoValues;
    delete[] defCoords;
}

i32 main()
{
    FoldedWmass();
    return 0;
}
