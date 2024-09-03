/// Example based on the df102_NanoAODDimuonAnalysis.C tutorial
/// Original: https://root.cern/doc/master/df102__NanoAODDimuonAnalysis_8C.html

#include <iostream>

// ROOT
#include <TFile.h>
#include <ROOT/RDF/RInterface.hxx>

#include "types.h"

#include "InvariantMass.h"

#define BINS 30000
#define BATCH_SIZE 32768
#define RUNS 3

void DiMuonGpu(Timer<> *rtTransfer, Timer<> *rtFill, Timer<> *rtResult) {
    TFile file("data/Run2012BC_DoubleMuParked_Muons.root");
    auto tree = dynamic_cast<TTree *>(file.Get("Events"));
    auto gpuHisto = GpuInvariantMassHisto(
        BINS + 2, 0.25, 300, 8, BATCH_SIZE,
        rtTransfer, rtFill, rtResult
    );
    auto coords = new f64[8 * BATCH_SIZE];
    auto *gpuResults = new f64[BINS + 2];

    // Values on stack
    u32 nMuon;
    i32 muonCharge[2];
    f32 pt[2], eta[2], phi[2], mass[2];

    tree->SetBranchAddress("nMuon", &nMuon);
    tree->SetBranchAddress("Muon_charge", muonCharge);
    tree->SetBranchAddress("Muon_pt", pt);
    tree->SetBranchAddress("Muon_eta", eta);
    tree->SetBranchAddress("Muon_phi", phi);
    tree->SetBranchAddress("Muon_mass", mass);

    const auto nMuonBranch = tree->GetBranch("nMuon");
    const auto chargeBranch = tree->GetBranch("Muon_charge");
    const auto ptBranch = tree->GetBranch("Muon_pt");
    const auto etaBranch = tree->GetBranch("Muon_eta");
    const auto phiBranch = tree->GetBranch("Muon_phi");
    const auto massBranch = tree->GetBranch("Muon_mass");

    // Process the batches
    usize offset = 0;
    isize nEntries = tree->GetEntries();
    for (isize i = 0; i < nEntries; ++i) {
        nMuonBranch->GetEntry(i);
        if (nMuon != 2) continue;

        chargeBranch->GetEntry(i);
        if (muonCharge[0] == muonCharge[1]) continue;

        ptBranch->GetEntry(i);
        etaBranch->GetEntry(i);
        phiBranch->GetEntry(i);
        massBranch->GetEntry(i);

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

    // Process the last batch
    if (offset != 0) {
        gpuHisto.FillN(offset / 8, coords);
    }

    // Retrieve the gpu results
    gpuHisto.RetrieveResults(gpuResults);
}

int main()
{
    Timer<> rtsTransfer[RUNS], rtsFill[RUNS], rtsResult[RUNS];
    for (usize i = 0; i < RUNS; ++i)
        DiMuonGpu(&rtsTransfer[i], &rtsFill[i], &rtsResult[i]);
    std::cerr << "Transfer      "; printTimerMinMaxAvg(rtsTransfer, RUNS);
    std::cerr << "Define + Fill "; printTimerMinMaxAvg(rtsFill, RUNS);
    std::cerr << "Result        "; printTimerMinMaxAvg(rtsResult, RUNS);

    return 0;
}
