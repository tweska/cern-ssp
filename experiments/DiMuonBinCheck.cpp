/// Example based on the df102_NanoAODDimuonAnalysis.C tutorial
/// Original: https://root.cern/doc/master/df102__NanoAODDimuonAnalysis_8C.html

#include <iostream>

// ROOT
#include <TFile.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>
#include <ROOT/RDF/RInterface.hxx>

#include "types.h"

#include "CpuFindBin.h"
#include "InvariantMass.h"

#define BINS 30000

int main()
{
    ROOT::EnableImplicitMT();

    auto coord = new f64[8];
    auto gpuHisto = GpuInvariantMassHisto(
        BINS + 2, 0.25, 300, 8, 1
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

    // Process the events
    while (reader.Next()) {
        if (*nMuon == 2 && muon_charge[0] != muon_charge[1]) {
            coord[0] = pt[0];
            coord[1] = pt[1];
            coord[2] = eta[0];
            coord[3] = eta[1];
            coord[4] = phi[0];
            coord[5] = phi[1];
            coord[6] = mass[0];
            coord[7] = mass[1];

            const auto gpuBin = gpuHisto.ExecuteOp(coord);

            const auto cpuInM = ROOT::VecOps::InvariantMasses<f64>(
                {pt[0]}, {eta[0]}, {phi[0]}, {mass[0]},
                {pt[1]}, {eta[1]}, {phi[1]}, {mass[1]}
            )[0];
            const auto cpuBin = CpuFindFixedBin(cpuInM, BINS, 0.25, 300);

            const auto deltaBin = std::abs(static_cast<isize>(cpuBin) - static_cast<isize>(gpuBin));
            if (deltaBin > 0) {
                std::cerr << "cpuBin=" << cpuBin << " gpuBin=" << gpuBin << " deltaBin=" << deltaBin << std::endl
                          << "pt0=" << pt[0] << " eta0=" << eta[0] << " phi0=" << phi[0] << " mass0=" << mass[0] << std::endl
                          << "pt1=" << pt[1] << " eta1=" << eta[1] << " phi1=" << phi[1] << " mass1=" << mass[1] << std::endl
                          << "TEST FAILED!" << std::endl;
                // return -1;
            }
        }
    }

    delete[] coord;
    return 0;
}
