#ifndef COORDS_H
#define COORDS_H

#include "types.h"

typedef struct
{
    f32 recoPt1; f32 recoEta1; f32 recoPhi1; f32 recoE1;
    f32 recoPt2; f32 recoEta2; f32 recoPhi2; f32 recoE2;
    f32 truePt1; f32 truePt2;  // May be negative to indicate invalid values!
} DefCoords;

void getCoords(DefCoords *coords, usize n);

#endif //COORDS_H
