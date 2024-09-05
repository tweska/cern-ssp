#include <vector>
#include <Math/Vector4D.h>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <TTreePerfStats.h>

#include "types.h"

using TLV = ROOT::Math::PtEtaPhiEVector;

f32 forwardFoldingFormula(
    const f32 reco_pt,
    const f32 truth_pt,
    const f32 s,
    const f32 r
) {
    return s * reco_pt + (reco_pt - truth_pt) * (r - s);
}

TLV applyFF(
    const TLV& jet,
    const f32 truth_pt,
    const f32 s,
    const f32 r
) {
    const f32 foldedPt = forwardFoldingFormula(jet.Pt(), truth_pt, s, r);
    TLV result(foldedPt, jet.Eta(), jet.Phi(), jet.E());

    return result;
}

TLV foldedWTLV(
    const std::vector<TLV>& reco_jets,
    const std::vector<f32>& truth_pt,
    const i32 index1,
    const i32 index2,
    const std::vector<f32>& reco_deltaR,
    const std::vector<i32>& truth_index,
    const std::vector<f32>& truth_deltaR,
    const f32 s,
    const f32 r,
    const f32 isolationCritical
) {
    if (index1 < 0 || index2 < 0) {
        return TLV(0,0,0,0);
    }

    const TLV& jet1 = reco_jets.at(index1);
    const TLV& jet2 = reco_jets.at(index2);

    const f32 recoIsol1 = reco_deltaR.at(index1);
    const f32 recoIsol2 = reco_deltaR.at(index2);

    if (recoIsol1 < isolationCritical || recoIsol2 < isolationCritical) {
        return jet1 + jet2;
    }

    const i32 index_truth1 = truth_index.at(index1);
    const i32 index_truth2 = truth_index.at(index2);

    if (index_truth1 < 0 || index_truth2 < 0) {
        return jet1 + jet2;
    }

    if (static_cast<u32>(index_truth1) >= truth_pt.size() || static_cast<u32>(index_truth2) >= truth_pt.size()) {
        return jet1 + jet2;
    }

    const f32 truth_pt1 = truth_pt.at(index_truth1);
    const f32 truth_pt2 = truth_pt.at(index_truth2);

    if (static_cast<u32>(index_truth1) >= truth_deltaR.size() || static_cast<u32>(index_truth2) >= truth_deltaR.size()) {
        return jet1 + jet2;
    }

    const f32 truthIsol1 = truth_deltaR.at(index_truth1);
    const f32 truthIsol2 = truth_deltaR.at(index_truth2);

    if (truthIsol1 < isolationCritical || truthIsol2 < isolationCritical) {
        return jet1 + jet2;
    }

    TLV folded1 = applyFF(jet1, truth_pt1, s, r);
    TLV folded2 = applyFF(jet2, truth_pt2, s, r);

    return folded1 + folded2;
}

void FoldedWmass()
{
    ROOT::EnableImplicitMT();

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

    auto createTLV = [](
        const std::vector<f32>& pt,
        const std::vector<f32>& eta,
        const std::vector<f32>& phi,
        const std::vector<f32>& e
    ) {
        std::vector<ROOT::Math::PtEtaPhiEVector> result;
        for (usize i = 0; i < pt.size(); ++i) {
            ROOT::Math::PtEtaPhiEVector vector(pt.at(i), eta.at(i), phi.at(i), e.at(i));
            result.emplace_back(vector);
        }
        return result;
    };
    auto node = df.Define(
        "jet_TLV_NOSYS",
        createTLV,
        {"jet_pt_NOSYS", "jet_eta", "jet_phi", "jet_e_NOSYS"}
    );

    std::vector<ROOT::RDF::RResultHandle> histos;
    for (i32 s = 0; s < 100; ++s) {
        for (i32 r = 0; r < 100; ++r) {
            const f32 scale = 0.9 + s*0.01;
            const f32 resolution = 0.8 + r*0.02;
            auto foldedWmass = [scale,resolution](
                const std::vector<TLV>& reco_jets,
                const std::vector<f32>& truth_pt,
                const i32 index1,
                const i32 index2,
                const std::vector<f32>& reco_deltaR,
                const std::vector<i32>& truth_index,
                const std::vector<f32>& truth_deltaR
            ) {
                TLV w = foldedWTLV(reco_jets, truth_pt, index1, index2, reco_deltaR, truth_index, truth_deltaR, scale, resolution, 0.5);
                return w.M()/1e3;
            };

            const std::string name = "folded_w_mass_GeV_s_" + std::to_string(s) + "_r_" + std::to_string(r) + "_isolation_0p5_NOSYS";
            auto newNode = node.Define(
                name,
                foldedWmass,
                {"jet_TLV_NOSYS", "particleLevel.jet_pt", "TtbarLjets_spanet_up_index_NOSYS", "TtbarLjets_spanet_down_index_NOSYS",
                 "jet_reco_to_reco_jet_closest_dR_NOSYS", "jet_truth_jet_paired_index", "jet_truth_to_truth_jet_closest_dR"});
            histos.emplace_back(newNode.Histo1D(name));
        }
    }
    RunGraphs(histos);

    treeStats->Print();
}

i32 main()
{
    FoldedWmass();
    return 0;
}
