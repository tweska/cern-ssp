/// Example based on the df102_NanoAODDimuonAnalysis.C tutorial
/// Original: https://root.cern/doc/master/df102__NanoAODDimuonAnalysis_8C.html

#include <iostream>
#include <chrono>

// ROOT
#include <TFile.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>
#include <ROOT/RDataFrame.hxx>

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
    ROOT::EnableImplicitMT();

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
                    gpuHisto.FillN(n, coords);
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

            gpuHisto.FillN(n, coords);
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
