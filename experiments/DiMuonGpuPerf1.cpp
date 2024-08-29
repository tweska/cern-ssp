/// Example based on the df102_NanoAODDimuonAnalysis.C tutorial
/// Original: https://root.cern/doc/master/df102__NanoAODDimuonAnalysis_8C.html

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
    auto coords = new f64[8 * BATCH_SIZE];
    auto *gpuResults = new f64[BINS + 2];

    for (usize i = 0; i < RUNS; ++i) {
        auto gpuHisto = GpuInvariantMassHisto(
            BINS + 2, 0.25, 300, 8, BATCH_SIZE
        );

        Timer runtimeSetup;
        Timer runtimeBatch;

        runtimeSetup.start();

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

        runtimeSetup.pause();
        runtimeBatch.start();

        // Process the batches
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
                    runtimeBatch.pause();
                    gpuHisto.FillN(BATCH_SIZE, coords);
                    runtimeBatch.start();
                    offset = 0;
                }
            }
        }

        // Process the last batch
        if (offset != 0) {
            gpuHisto.FillN(offset / 8, coords);
        }
        runtimeBatch.pause();

        // Retrieve the gpu results
        gpuHisto.RetrieveResults(gpuResults);

        // Print the runtimes
        gpuHisto.PrintRuntime();
        std::cout << "runtimeCpuSetup = " << std::setw(10) << std::fixed << std::setprecision(3) << runtimeSetup.getTotal() << std::endl;
        std::cout << "runtimeCpuBatch = " << std::setw(10) << std::fixed << std::setprecision(3) << runtimeBatch.getTotal() << std::endl << std::endl;
    }

    delete[] coords;
    delete[] gpuResults;
    return 0;
}
