/// Example based on the df102_NanoAODDimuonAnalysis.C tutorial
/// Original: https://root.cern/doc/master/df102__NanoAODDimuonAnalysis_8C.html

#include <chrono>
#include <iostream>
#include <iomanip>

// ROOT
#include <TFile.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>

#include "types.h"

#include "InvariantMass.h"

#define BINS 30000
#define BATCH_SIZE 1024
#define RUNS 10

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

struct runtime {
    f64 gpuInit;
    f64 gpuTransfer;
    f64 gpuKernel;
    f64 gpuResult;
    f64 cpuSetup;
    f64 cpuBatch;
};

int main()
{
    auto runtimes = new runtime[RUNS];
    auto coords = new f64[8 * BATCH_SIZE];
    auto *gpuResults = new f64[BINS + 2];

    for (usize i = 0; i < RUNS; ++i) {
        auto gpuHisto = GpuInvariantMassHisto(
            BINS + 2, 0.25, 300, 8, BATCH_SIZE
        );

        auto t0 = high_resolution_clock::now();

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

        auto t1 = high_resolution_clock::now();

        // Process the batches
        // usize n = 0;
        usize offset = 0;
        while (reader.Next()) {
            if (*nMuon == 2 && muon_charge[0] != muon_charge[1]) {
                coords[offset++] = pt[0];
                coords[offset++] = pt[1];
                coords[offset++] = eta[0];
                coords[offset++] = eta[1];
                coords[offset++] = phi[0];
                coords[offset++] = phi[1];
                coords[offset++] = mass[0];
                coords[offset++] = mass[1];

                if (offset == 8 * BATCH_SIZE) {
                    gpuHisto.FillN(BATCH_SIZE, coords);
                    offset = 0;
                }
            }
        }

        // Process the last batch
        if (offset != 0) {
            gpuHisto.FillN(offset / 8, coords);
        }

        auto t2 = high_resolution_clock::now();

        // Retreive the gpu results
        gpuHisto.RetrieveResults(gpuResults);

        // Save the runtimes
        gpuHisto.GetRuntime(
            &runtimes[i].gpuInit,
            &runtimes[i].gpuTransfer,
            &runtimes[i].gpuKernel,
            &runtimes[i].gpuResult
        );
        duration<f64, std::milli> runtimeSetup = t1 - t0;
        duration<f64, std::milli> runtimeBatch = t2 - t1;
        runtimes[i].cpuSetup = runtimeSetup.count();
        runtimes[i].cpuBatch = runtimeBatch.count() - (runtimes[i].gpuTransfer + runtimes[i].gpuKernel);

        // Print the runtimes
        gpuHisto.PrintRuntime();
        std::cout << "runtimeCpuSetup = " << std::setw(10) << std::fixed << std::setprecision(5) << runtimes[i].cpuSetup << std::endl;
        std::cout << "runtimeCpuBatch = " << std::setw(10) << std::fixed << std::setprecision(5) << runtimes[i].cpuBatch << std::endl << std::endl;
    }

    delete[] runtimes;
    delete[] coords;
    delete[] gpuResults;
    return 0;
}
