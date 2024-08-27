/// Example based on the df102_NanoAODDimuonAnalysis.C tutorial
/// Original: https://root.cern/doc/master/df102__NanoAODDimuonAnalysis_8C.html

#include <iostream>

// ROOT
#include <TFile.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>
#include <ROOT/RDataFrame.hxx>

#include "types.h"
#include "util.h"

#include "InvariantMass.h"

#define BINS 30000
#define BATCH_SIZE 1024

int main()
{
    ROOT::EnableImplicitMT();

    auto coords = new f64[8 * BATCH_SIZE];
    auto gpuHisto = GpuInvariantMassHisto(
        BINS + 2, 0.25, 300, 8, BATCH_SIZE
    );

    TFile file("data/Run2012BC_DoubleMuParked_Muons.root");
    TTreeReader reader("Events", &file);

    // Values for filtering
    TTreeReaderValue<u32> nMuon(reader, "nMuon");
    TTreeReaderArray<i32> muon_charge(reader, "Muon_charge");

    // Muon properties
    TTreeReaderArray<f32> pt(reader, "Muon_pt");
    TTreeReaderArray<f32> eta(reader, "Muon_eta");
    TTreeReaderArray<f32> phi(reader, "Muon_phi");
    TTreeReaderArray<f32> mass(reader, "Muon_mass");

    // Process the batches
    usize n = 0;
    while (reader.Next()) {
        if (*nMuon == 2 && muon_charge[0] != muon_charge[1]) {
            coords[n + 0 * BATCH_SIZE] = pt[0];
            coords[n + 1 * BATCH_SIZE] = pt[1];
            coords[n + 2 * BATCH_SIZE] = eta[0];
            coords[n + 3 * BATCH_SIZE] = eta[1];
            coords[n + 4 * BATCH_SIZE] = phi[0];
            coords[n + 5 * BATCH_SIZE] = phi[1];
            coords[n + 6 * BATCH_SIZE] = mass[0];
            coords[n + 7 * BATCH_SIZE] = mass[1];

            if (++n == BATCH_SIZE) {
                gpuHisto.FillN(n, coords, nullptr);
                n = 0;
            }
        }
    }

    // Process the last batch
    if (n > 0) {
        memcpy(&coords[1 * n], &coords[1 * BATCH_SIZE], n * sizeof(f64));
        memcpy(&coords[2 * n], &coords[2 * BATCH_SIZE], n * sizeof(f64));
        memcpy(&coords[3 * n], &coords[3 * BATCH_SIZE], n * sizeof(f64));
        memcpy(&coords[4 * n], &coords[4 * BATCH_SIZE], n * sizeof(f64));
        memcpy(&coords[5 * n], &coords[5 * BATCH_SIZE], n * sizeof(f64));
        memcpy(&coords[6 * n], &coords[6 * BATCH_SIZE], n * sizeof(f64));
        memcpy(&coords[7 * n], &coords[7 * BATCH_SIZE], n * sizeof(f64));

        gpuHisto.FillN(n, coords, nullptr);
    }

    // Retreive the gpu results
    f64 *gpuResults = new f64[BINS + 2];
    gpuHisto.RetrieveResults(gpuResults);


    // Same analysis on the CPU using RDF
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
    auto cpuHisto = df_mass.Histo1D({"Dimuon_mass", "Dimuon mass;m_{#mu#mu} (GeV);N_{Events}", BINS, 0.25, 300}, "Dimuon_mass");
    f64 *cpuResults = cpuHisto->GetArray();

    f64 maxError, gpuValue, cpuValue;
    arrayMaxError(&maxError, &gpuValue, &cpuValue, BINS + 2, gpuResults, cpuResults);
    std::cout << "Observed maximum error: " << maxError << std::endl;
    std::cout << "GPU value: " << gpuValue << std::endl;
    std::cout << "CPU value: " << cpuValue << std::endl;

    delete[] coords;
    delete[] gpuResults;
    return 0;
}
