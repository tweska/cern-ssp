/// Example based on the df102_NanoAODDimuonAnalysis.C tutorial
/// Original: https://root.cern/doc/master/df102__NanoAODDimuonAnalysis_8C.html

#include <iostream>

// ROOT
#include <thread>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RLogger.hxx>

#include "types.h"
#include "timer.h"

#define BINS 30000
#define THREADS 16
#define BATCH_SIZE (2048 * THREADS)
#define RUNS 10

void bulkThread(const f64 *coords, TH1D *histo, const usize n, const usize tid) {
    const usize start = BATCH_SIZE / THREADS * tid;
    const usize end = std::min(start + BATCH_SIZE / THREADS, n);
    usize offset = start * 8;

    for (usize i = start; i < end; ++i) {
        const f64 invariantMass = ROOT::VecOps::InvariantMasses<f64>(
            {coords[offset + 0]}, {coords[offset + 2]}, {coords[offset + 4]}, {coords[offset + 6]},
            {coords[offset + 1]}, {coords[offset + 3]}, {coords[offset + 5]}, {coords[offset + 7]}
        )[0];
        histo->Fill(invariantMass);
        offset += 8;
    }
}

void processBulk(f64 *coords, std::vector<TH1D*> histos, usize n) {
    std::vector<std::thread> threads;
    threads.reserve(THREADS);
    for (usize tid = 0; tid < THREADS; ++tid) {
        threads.emplace_back(bulkThread, coords, histos[tid], n, tid);
    }

    for (auto &thread : threads) {
        thread.join();
    }
}

void DiMuonCpu(Timer<> &rtCombined) {
    TFile file("data/Run2012BC_DoubleMuParked_Muons.root");
    auto tree = dynamic_cast<TTree *>(file.Get("Events"));
    std::vector<TH1D*> histos;
    histos.reserve(THREADS);
    for (usize tid = 0; tid < THREADS; ++tid) {
        histos.push_back(new TH1D(
            ("Dimuon_mass_" + std::to_string(tid)).c_str(),
            "Dimuon mass;m_{#mu#mu} (GeV);N_{Events}",
            BINS, 0.25, 300
        ));
    }
    auto coords = new f64[8 * BATCH_SIZE];

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
            rtCombined.start();
            processBulk(coords, histos, BATCH_SIZE);
            rtCombined.pause();
            offset = 0;
        }
    }

    // Process the last batch
    if (offset != 0) {
        rtCombined.start();
        processBulk(coords, histos, offset / 8);
        rtCombined.pause();
    }

    TH1D mergedHisto(
        "Merged_Dimuon_mass",
        "Merged Dimuon mass;m_{#mu#mu} (GeV);N_{Events}",
        BINS, 0.25, 300
    );
    for (const auto &histo : histos) {
        mergedHisto.Add(histo);
    }

    f64 *results = mergedHisto.GetArray();
    (void) results;

    delete[] coords;
}

int main()
{
    Timer<> rtsCombined[RUNS];
    for (usize i = 0; i < RUNS; ++i)
        DiMuonCpu(rtsCombined[i]);
    std::cerr << "Define + Fill "; printTimerMinMaxAvg(rtsCombined, RUNS);

    return 0;
}
