/// Example based on the df102_NanoAODDimuonAnalysis.C tutorial
/// Original: https://root.cern/doc/master/df102__NanoAODDimuonAnalysis_8C.html

#include <chrono>
#include <iostream>
#include <iomanip>

// ROOT
#include <ROOT/RDataFrame.hxx>

#include "types.h"

#define RUNS 10

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

int main()
{
    ROOT::EnableImplicitMT();

    for (usize i = 0; i < RUNS; ++i) {
        auto t0 = high_resolution_clock::now();

        ROOT::RDataFrame df("Events", "~/root-files/Run2012BC_DoubleMuParked_Muons.root");
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

        auto t1 = high_resolution_clock::now();
        duration<f64, std::milli> runtime = t1 - t0;
        std::cout << "runtime = " << std::setw(10) << std::fixed << std::setprecision(5) << runtime.count() << std::endl;
    }

    return 0;
}
