#include "test.h"

int main(int argc, char **argv)
{
    auto test = Single1DFixedUniformHistogramTest(
        10 + 2,  // nBins
        0.0,     // xMin
        1.0,     // xMax
        5000000  // BulkSize
    );
    test.fullTest();

    return 0;
}
