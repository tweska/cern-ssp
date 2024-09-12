#ifndef CPUFWM_H
#define CPUFWM_H

#include "types.h"

f32 foldedMass(
    f32 recoPt1, f32 recoEta1, f32 recoPhi1, f32 recoE1,
    f32 recoPt2, f32 recoEta2, f32 recoPhi2, f32 recoE2,
    f32 truePt1, f32 truePt2,
    f32 scale, f32 resolution
);

#endif //CPUFWM_H
