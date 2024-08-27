#ifndef CPUFINDBIN_H
#define CPUFINDBIN_H

#include "types.h"

inline usize CpuFindFixedBin(f64 x, usize nBins, f64 xMin, f64 xMax)
{
    if (x < xMin)
        return 0;
    if (!(x < xMax))
        return nBins + 1;

    return 1 + static_cast<usize>(nBins * (x - xMin) / (xMax - xMin));
}

#endif //CPUFINDBIN_H
