#include <vector>

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <TTreePerfStats.h>

#include "types.h"

#define ISOLATION_CRITICAL 0.5

inline f32 angle(
    const f32 x1, const f32 y1, const f32 z1,
    const f32 x2, const f32 y2, const f32 z2
) {
    // cross product
    const f32 cx = y1 * z2 - y2 * z1;
    const f32 cy = x1 * z2 - x2 * z1;
    const f32 cz = x1 * y2 - x2 * y1;

    // norm of cross product
    const f32 c = std::sqrt(cx * cx + cy * cy + cz * cz);

    // dot product
    const f32 d = x1 * x2 + y1 * y2 + z1 * z2;

    return std::atan2(c, d);
}

inline f32 invariantMassPxPyPzM(
   const f32 x1, const f32 y1, const f32 z1, const f32 mass1,
   const f32 x2, const f32 y2, const f32 z2, const f32 mass2
) {
    // Numerically stable computation of Invariant Masses
    const f32 pp1 = x1 * x1 + y1 * y1 + z1 * z1;
    const f32 pp2 = x2 * x2 + y2 * y2 + z2 * z2;

    if (pp1 <= 0 && pp2 <= 0)
        return (mass1 + mass2);
    if (pp1 <= 0) {
        f32 mm = mass1 + std::sqrt(mass2*mass2 + pp2);
        f32 m2 = mm*mm - pp2;
        return m2 >= 0 ? std::sqrt(m2) : std::sqrt(-m2);
    }
    if (pp2 <= 0) {
        f32 mm = mass2 + std::sqrt(mass1*mass1 + pp1);
        f32 m2 = mm*mm - pp1;
        return m2 >= 0 ? std::sqrt(m2) : std::sqrt(-m2);
    }

    const f32 mm1 =  mass1 * mass1;
    const f32 mm2 =  mass2 * mass2;

    const f32 r1 = mm1 / pp1;
    const f32 r2 = mm2 / pp2;
    const f32 x = r1 + r2 + r1 * r2;
    const f32 a = angle(x1, y1, z1, x2, y2, z2);
    const f32 cos_a = std::cos(a);
    f32 y;
    if (cos_a >= 0){
        y = (x + std::sin(a) * std::sin(a)) / (std::sqrt(x + 1) + cos_a);
    } else {
        y = std::sqrt(x + 1) - cos_a;
    }

    const f32 z = 2.0f * std::sqrt(pp1 * pp2);

    // Return invariant mass with (+, -, -, -) metric
    return std::sqrt(mm1 + mm2 + y * z);
}

inline f32 invariantMassPxPyPzE(
   const f32 x1, const f32 y1, const f32 z1, const f32 e1,
   const f32 x2, const f32 y2, const f32 z2, const f32 e2
) {
    const f32 pp1 = x1 * x1 + y1 * y1 + z1 * z1;
    const f32 pp2 = x2 * x2 + y2 * y2 + z2 * z2;

    const f32 mm1 = e1 * e1 - pp1;
    const f32 mm2 = e2 * e2 - pp2;

    const f32 mass1 = (mm1 >= 0) ? std::sqrt(mm1) : 0;
    const f32 mass2 = (mm2 >= 0) ? std::sqrt(mm2) : 0;

    return invariantMassPxPyPzM(x1, y1, z1, mass1, x2, y2, z2, mass2);
}

inline f32 invariantMassPtEtaPhiE(
    const f32 pt1, const f32 eta1, const f32 phi1, const f32 e1,
    const f32 pt2, const f32 eta2, const f32 phi2, const f32 e2
) {
    const f32 x1 = pt1 * std::cos(phi1);
    const f32 y1 = pt1 * std::sin(phi1);
    const f32 z1 = pt1 * std::sinh(eta1);

    const f32 x2 = pt2 * std::cos(phi2);
    const f32 y2 = pt2 * std::sin(phi2);
    const f32 z2 = pt2 * std::sinh(eta2);

    return invariantMassPxPyPzE(x1, y1, z1, e1, x2, y2, z2, e2);
}

inline f32 forwardFolding(
    const f32 recoPt,
    const f32 truePt,
    const f32 s,
    const f32 r
) {
    return s * recoPt + (recoPt - truePt) * (r - s);
}

f32 foldedMass(
    f32 recoPt1, const f32 recoEta1, const f32 recoPhi1, const f32 recoE1,
    f32 recoPt2, const f32 recoEta2, const f32 recoPhi2, const f32 recoE2,
    const f32 truePt1, const f32 truePt2,
    const f32 scale, const f32 resolution,
    const b8 foldable
) {
    if (foldable) {
        recoPt1 = forwardFolding(recoPt1, truePt1, scale, resolution);
        recoPt2 = forwardFolding(recoPt2, truePt2, scale, resolution);
    }

    // Return Invariant mass of sum.
    return invariantMassPtEtaPhiE(
        recoPt1, recoEta1, recoPhi1, recoE1,
        recoPt2, recoEta2, recoPhi2, recoE2
    ) / 1e3f;
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

    std::vector<ROOT::RDF::RResultHandle> histos;
    for (i32 s = 0; s < 100; ++s) {
        for (i32 r = 0; r < 100; ++r) {
            const f32 scale = 0.9 + s*0.01;
            const f32 resolution = 0.8 + r*0.02;
            auto foldedWmass = [scale,resolution](
                const std::vector<f32>& recoPt,
                const std::vector<f32>& recoEta,
                const std::vector<f32>& recoPhi,
                const std::vector<f32>& recoE,
                const std::vector<f32>& truePt,
                const i32 i1, const i32 i2,
                const std::vector<f32>& recoDeltaR,
                const std::vector<i32>& trueI,
                const std::vector<f32>& trueDeltaR
            ) {
                f32 recoIsol1, recoIsol2;
                i32 trueI1,    trueI2;
                f32 truePt1,   truePt2;
                f32 trueIsol1, trueIsol2;

                recoIsol1 = recoDeltaR[i1];
                recoIsol2 = recoDeltaR[i2];
                if (recoIsol1 < ISOLATION_CRITICAL || recoIsol2 < ISOLATION_CRITICAL) {
                    goto unfoldable;
                }

                trueI1 = trueI[i1];
                trueI2 = trueI[i2];
                if (trueI1 < 0 || trueI2 < 0) {
                    goto unfoldable;
                }
                if (static_cast<u32>(trueI1) >= truePt.size() || static_cast<u32>(trueI2) >= truePt.size()) {
                    goto unfoldable;
                }

                truePt1 = truePt[trueI1];
                truePt2 = truePt[trueI2];
                if (static_cast<u32>(trueI1) >= trueDeltaR.size() || static_cast<u32>(trueI2) >= trueDeltaR.size()) {
                    goto unfoldable;
                }

                trueIsol1 = trueDeltaR[trueI1];
                trueIsol2 = trueDeltaR[trueI2];
                if (trueIsol1 < ISOLATION_CRITICAL || trueIsol2 < ISOLATION_CRITICAL) {
                    goto unfoldable;
                }

                return foldedMass(
                    recoPt[i1], recoEta[i1], recoPhi[i1], recoE[i1],
                    recoPt[i2], recoEta[i2], recoPhi[i2], recoE[i2],
                    truePt1, truePt2,
                    scale, resolution,
                    true
                );

                unfoldable:
                return foldedMass(
                    recoPt[i1], recoEta[i1], recoPhi[i1], recoE[i1],
                    recoPt[i2], recoEta[i2], recoPhi[i2], recoE[i2],
                    0, 0,
                    scale, resolution,
                    false
                );
            };

            const std::string name = "folded_w_mass_GeV_s_" + std::to_string(s) + "_r_" + std::to_string(r) + "_isolation_0p5_NOSYS";
            auto newNode = df.Define(
                name,
                foldedWmass,
                {"jet_pt_NOSYS", "jet_eta", "jet_phi", "jet_e_NOSYS", "particleLevel.jet_pt",
                 "TtbarLjets_spanet_up_index_NOSYS", "TtbarLjets_spanet_down_index_NOSYS",
                 "jet_reco_to_reco_jet_closest_dR_NOSYS", "jet_truth_jet_paired_index",
                 "jet_truth_to_truth_jet_closest_dR"}
            );
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
