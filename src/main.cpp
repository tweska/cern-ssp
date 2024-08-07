#include "util.h"
#include "GHisto.h"
#include "GbHisto.h"

int main(int argc, char **argv) {
    // Single histogram
    std::cout << std::endl << "=== SINGLE HISTOGRAM ===" << std::endl;
    GHisto<double, 1> *histo = new GHisto<double, 1>(
        new u32[1] {10},
        new f64[1] {0}, new f64[1] {10},
        nullptr, new i32[1] {-1}
    );

    histo->Fill(5, new f64[5] {0.42, 8.2, 42.0, 8.3, 6.5});
    f64 histogram[10];
    histo->RetrieveResults(histogram, nullptr);
    printArray(histogram, 10);


    // Batch of one histogram
    std::cout << std::endl << "=== ONE HISTOGRAM BATCH ===" << std::endl;
    GbHisto<f64> *histos1 = new GbHisto<f64>(
        1, new u32[1] {1}, new u32[1] {10},
        new f64[1] {0}, new f64[1] {10},
        nullptr, new i32[1] {-1}
    );

    histos1->Fill(5, new f64[5] {0.42, 8.2, 42.0, 8.3, 6.5});
    f64 *histograms1 = new f64[10];
    histos1->RetrieveResults(histograms1, nullptr);
    printArray(histograms1, 10);
    delete histograms1;


    // Batch of two histograms
    std::cout << std::endl << "=== TWO HISTOGRAM BATCH ===" << std::endl;
    GbHisto<f64> *histos2 = new GbHisto<f64>(
        2, new u32[2] {1, 1}, new u32[2] {10, 10},
        new f64[2] {0, 0}, new f64[2] {10, 10},
        nullptr, new i32[2] {-1, -1}
    );

    histos2->Fill(5, new f64[10] {
        0.42, 8.2, 42.0, 8.3, 6.5,
        0.42, 8.2, 42.0, 8.3, 6.5
    });
    f64 *histograms2 = new f64[20];
    histos2->RetrieveResults(histograms2, nullptr);
    printArray(histograms2, 10);
    printArray(&histograms2[10], 10);
    delete histograms2;

    return 0;
}
