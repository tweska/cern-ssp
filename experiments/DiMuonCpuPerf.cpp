/// Example based on the df102_NanoAODDimuonAnalysis.C tutorial
/// Original: https://root.cern/doc/master/df102__NanoAODDimuonAnalysis_8C.html

#include <iostream>

// ROOT
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RLogger.hxx>

#include "types.h"
#include "timer.h"

#define BINS 30000
#define RUNS 3

void DiMuonCpu(Timer<> &rtDefine, Timer<> &rtFill) {
    TFile file("data/Run2012BC_DoubleMuParked_Muons.root");
    auto tree = dynamic_cast<TTree *>(file.Get("Events"));
    auto histo = TH1D(
        "Dimuon_mass",
        "Dimuon mass;m_{#mu#mu} (GeV);N_{Events}",
        BINS, 0.25, 300
    );

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

        rtDefine.start();
        f64 invariantMass = ROOT::VecOps::InvariantMasses<f64>(
            {pt[0]}, {eta[0]}, {phi[0]}, {mass[0]},
            {pt[1]}, {eta[1]}, {phi[1]}, {mass[1]}
        )[0];
        rtDefine.pause();
        rtFill.start();
        histo.Fill(invariantMass);
        rtFill.pause();
    }

    f64 *results = histo.GetArray();
    (void) results;
}

int main()
{
    Timer<> rtsDefine[RUNS], rtsFill[RUNS];
    for (usize i = 0; i < RUNS; ++i)
        DiMuonCpu(rtsDefine[i], rtsFill[i]);
    std::cerr << "Define        "; printTimerMinMaxAvg(rtsDefine, RUNS);
    std::cerr << "Fill          "; printTimerMinMaxAvg(rtsFill, RUNS);

    Timer<> rtsBoth[RUNS];
    for (usize i = 0; i < RUNS; ++i)
        rtsBoth[i] = rtsDefine[i] + rtsFill[i];
    std::cerr << "Define + Fill "; printTimerMinMaxAvg(rtsBoth, RUNS);

    return 0;
}
