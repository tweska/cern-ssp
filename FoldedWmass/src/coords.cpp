#include "types.h"

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDFHelpers.hxx>

#include "coords.h"

#define ISOLATION_CRITICAL 0.5

void getCoords(DefCoords *coords, usize n) {
    TChain* chainReco = new TChain("reco");
    TChain* chainTruth = new TChain("particleLevel");

    chainReco->AddFile("data/output.root");
    chainTruth->AddFile("data/output.root");
    chainTruth->BuildIndex("eventNumber");
    chainReco->AddFriend(chainTruth);

    auto df = ROOT::RDataFrame(*chainReco).Filter(
        [](const i32 upIndex, i32 const downIndex) { return upIndex >= 0 && downIndex >= 0; },
        {"TtbarLjets_spanet_up_index_NOSYS", "TtbarLjets_spanet_down_index_NOSYS"}
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

    usize i = 0;
    df.Foreach(
        [&i, n, coords] (
        const std::vector<f32>& recoPt,
        const std::vector<f32>& recoEta,
        const std::vector<f32>& recoPhi,
        const std::vector<f32>& recoE,
        const i32 i1, const i32 i2,
        const f32 truePt1, const f32 truePt2
    ) {
        if (i >= n) { return; }
        coords[i++] = {
            recoPt[i1], recoEta[i1], recoPhi[i1], recoE[i1],
            recoPt[i2], recoEta[i2], recoPhi[i2], recoE[i2],
            truePt1, truePt2
        };
    },
    {
        "jet_pt_NOSYS", "jet_eta", "jet_phi", "jet_e_NOSYS",
        "TtbarLjets_spanet_up_index_NOSYS", "TtbarLjets_spanet_down_index_NOSYS",
        "truePt1", "truePt2"
    });

    while (i < n) {
        const usize len = std::min(i, n - i);
        memcpy(&coords[i], coords, len * sizeof(DefCoords));
        i += len;
    }
}