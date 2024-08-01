#include "util.h"
#include "GHisto.h"

int main(int argc, char **argv) {
    GHisto<double, 1, 256> *histo = new GHisto<double, 1, 256>(
        new u32[1] {100},
        new f64[1] {0}, new f64[1] {100},
        nullptr, new i32[1] {-1}
    );

    histo->Fill(5, new double[5] {0.42, 8.2, 42.0, 8.3, 6.5});

    double histogram[100];
    histo->RetrieveResults(histogram, nullptr);
    printArray(histogram, 100);

    return 0;
}
