/// Example based on the df102_NanoAODDimuonAnalysis.C tutorial
/// Original: https://root.cern/doc/master/df102__NanoAODDimuonAnalysis_8C.html

#include <iostream>
#include <iomanip>

// ROOT
#include <ROOT/RDataFrame.hxx>

#include "types.h"
#include "timer.h"
#include "InvariantMass.h"

#define BINS 30000
#define BATCH_SIZE 1024
#define RUNS 10

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

        ROOT::RDataFrame df("Events", "data/Run2012BC_DoubleMuParked_Muons.root");
        auto df_os = df.Filter("nMuon == 2")
                       .Filter("Muon_charge[0] != Muon_charge[1]");

        runtimeSetup.pause();
        runtimeBatch.start();

        // Process the batches
        usize offset = 0;
        df_os.Foreach([&offset, coords, &gpuHisto](ROOT::RVec<f32> pt, ROOT::RVec<f32> eta, ROOT::RVec<f32> phi, ROOT::RVec<f32> mass) {
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
        },
        {"Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass"});

        // Process the last batch
        if (offset != 0) {
            gpuHisto.FillN(offset / 8, coords);
        }
        runtimeBatch.pause();

        // Retreive the gpu results
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
