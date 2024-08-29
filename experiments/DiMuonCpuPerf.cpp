/// Example based on the df102_NanoAODDimuonAnalysis.C tutorial
/// Original: https://root.cern/doc/master/df102__NanoAODDimuonAnalysis_8C.html

#include <iostream>

// ROOT
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RLogger.hxx>

#include "types.h"
#include "timer.h"

#define RUNS 10

int main()
{
    Timer timer;

    ROOT::EnableImplicitMT();
    auto verbosity = ROOT::Experimental::RLogScopedVerbosity(
        ROOT::Detail::RDF::RDFLogChannel(),
        ROOT::Experimental::ELogLevel::kInfo
    );

    for (usize i = 0; i < RUNS; ++i) {
        timer.start();

        ROOT::RDataFrame df("Events", "data/Run2012BC_DoubleMuParked_Muons.root");
        auto df_os = df.Filter("nMuon == 2")
                       .Filter("Muon_charge[0] != Muon_charge[1]");
        auto df_mass = df_os.Define("Dimuon_mass",
                                    [](const ROOT::RVec<f32>& pt, const ROOT::RVec<f32>& eta,
                                       const ROOT::RVec<f32>& phi, const ROOT::RVec<f32>& mass) {
                                        return ROOT::VecOps::InvariantMasses<f64>(
                                            {pt[0]}, {eta[0]}, {phi[0]}, {mass[0]},
                                            {pt[1]}, {eta[1]}, {phi[1]}, {mass[1]});},
                                            {"Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass"});
        auto cpuHisto = df_mass.Histo1D({"Dimuon_mass", "Dimuon mass;m_{#mu#mu} (GeV);N_{Events}", 30000, 0.25, 300}, "Dimuon_mass");
        f64 *cpuResults = cpuHisto->GetArray();
        (void) cpuResults;

        timer.pause();
        std::cout << "runtime = " << std::setw(10) << std::fixed << std::setprecision(3) << timer.getTotal() << std::endl;
    }

    return 0;
}
