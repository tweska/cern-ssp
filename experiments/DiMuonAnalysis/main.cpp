#include <chrono>
#include <iostream>

// ROOT
#include <TFile.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>
#include <ROOT/RDataFrame.hxx>

#include "GHisto.h"
#include "../../inc/types.h"
#include "../../inc/util.h"

#define MAX_ERROR 0.01
#define BATCH_SIZE 1024
#define BINS 30000

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

i32 main() {
    ROOT::EnableImplicitMT();
    auto t0 = high_resolution_clock::now();

    f64 *coords = new f64[8 * BATCH_SIZE];
    auto gpuHisto = GHistoIM(BINS + 2, 0.25, 300, 8, BATCH_SIZE);

    TFile file("~/root-files/Run2012BC_DoubleMuParked_Muons.root");
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

    auto t1 = high_resolution_clock::now();

    // Same analysis on the CPU using RDF
    ROOT::RDataFrame df("Events", "~/root-files/Run2012BC_DoubleMuParked_Muons.root");
    auto df_os = df.Filter("nMuon == 2")
                   .Filter("Muon_charge[0] != Muon_charge[1]");
    auto df_mass = df_os.Define("Dimuon_mass",
                                ROOT::VecOps::InvariantMass<f32>,
                                {"Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass"});
    auto cpuHisto = df_mass.Histo1D({"Dimuon_mass", "Dimuon mass;m_{#mu#mu} (GeV);N_{Events}", BINS, 0.25, 300}, "Dimuon_mass");
    f64 *cpuResults = cpuHisto->GetArray();

    auto t2 = high_resolution_clock::now();

    duration<f64, std::milli> gpuRuntimeMs = t1 - t0;
    duration<f64, std::milli> cpuRuntimeMs = t2 - t1;
    std::cout << "Timing results:" << std::endl;
    std::cout << "    GPU: " << gpuRuntimeMs.count() << std::endl;
    std::cout << "    CPU: " << cpuRuntimeMs.count() << std::endl;

    printArray(gpuResults, 16);
    printArray(cpuResults, 16);

    if (checkArray(BINS + 2, gpuResults, cpuResults, MAX_ERROR)) {
        std::cout << "TEST SUCCEEDED!" << std::endl;
    } else {
        std::cout << "TEST FAILED!" << std::endl;
    }

    delete[] coords;
    delete[] gpuResults;
    return 0;
}
