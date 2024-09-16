#ifndef INVARIANTMASS_H
#define INVARIANTMASS_H

#include "types.h"

#include "GpuDefHisto.h"

f64 InvariantMass(f64 *coords, usize i, usize n);
using GpuInvariantMassHisto = GpuDefHisto<f64, InvariantMass>;

#endif //INVARIANTMASS_H
