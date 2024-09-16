#include <vector>

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <TTreePerfStats.h>

#include "types.h"
#include "util.h"
#include "CpuFWM.h"

#define ISOLATION_CRITICAL 0.5

#define NBINS 100
#define XMIN    0
#define XMAX  400

void FoldedWmass(b8 print = false)
{
    ROOT::EnableImplicitMT();

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

    std::vector<ROOT::RDF::RResultHandle> histos;
    for (i32 s = 0; s < 100; ++s) {
        for (i32 r = 0; r < 100; ++r) {
            const f32 scale = 0.9f + static_cast<f32>(s) * 0.01f;
            const f32 resolution = 0.8f + static_cast<f32>(r) * 0.02f;
            auto foldedWmass = [scale,resolution](
                const std::vector<f32>& recoPt,
                const std::vector<f32>& recoEta,
                const std::vector<f32>& recoPhi,
                const std::vector<f32>& recoE,
                const i32 i1, const i32 i2,
                const f32 truePt1, const f32 truePt2
            ) {
                return static_cast<f64>(foldedMass(
                    recoPt[i1], recoEta[i1], recoPhi[i1], recoE[i1],
                    recoPt[i2], recoEta[i2], recoPhi[i2], recoE[i2],
                    truePt1, truePt2,
                    scale, resolution
                ));
            };

            const std::string name = "folded_w_mass_GeV_s_" + std::to_string(s) + "_r_" + std::to_string(r) + "_isolation_0p5_NOSYS";
            auto newNode = df.Define(
                name,
                foldedWmass,
                {
                    "jet_pt_NOSYS", "jet_eta", "jet_phi", "jet_e_NOSYS",
                    "TtbarLjets_spanet_up_index_NOSYS", "TtbarLjets_spanet_down_index_NOSYS",
                    "truePt1", "truePt2"
                }
            );
            histos.emplace_back(newNode.Histo1D({name.c_str(), "", NBINS, XMIN, XMAX}, name));
        }
    }
    RunGraphs(histos);

    if (print) {
        for (int s = 0; s < 100; ++s) {
            for (int r = 0; r < 100; ++r) {
                std::cout << "r=" << std::setw(2) << r << " s=" << std::setw(2) << s << " : ";
                printArray(histos[r * 100 + s].GetPtr<TH1D>()->GetArray(), NBINS + 2);
            }
        }
    }
}

i32 main(i32 argc, c8 *argv[])
{
    b8 printFlag = false;
    for (i32 i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--print") == 0) { printFlag = true; }
    }

    FoldedWmass(printFlag);
    return 0;
}
