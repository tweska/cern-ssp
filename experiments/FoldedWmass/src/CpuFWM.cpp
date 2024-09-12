#include <cmath>

#include "types.h"

#include "CpuFWM.h"

#define ISOLATION_CRITICAL 0.5

inline f32 angle(
    const f32 x1, const f32 y1, const f32 z1,
    const f32 x2, const f32 y2, const f32 z2
) {
    // cross product
    const f32 cx = y1 * z2 - y2 * z1;
    const f32 cy = x1 * z2 - x2 * z1;
    const f32 cz = x1 * y2 - x2 * y1;

    // norm of cross product
    const f32 c = std::sqrt(cx * cx + cy * cy + cz * cz);

    // dot product
    const f32 d = x1 * x2 + y1 * y2 + z1 * z2;

    return std::atan2(c, d);
}

inline f32 invariantMassPxPyPzM(
   const f32 x1, const f32 y1, const f32 z1, const f32 mass1,
   const f32 x2, const f32 y2, const f32 z2, const f32 mass2
) {
    // Numerically stable computation of Invariant Masses
    const f32 pp1 = x1 * x1 + y1 * y1 + z1 * z1;
    const f32 pp2 = x2 * x2 + y2 * y2 + z2 * z2;

    if (pp1 <= 0 && pp2 <= 0)
        return (mass1 + mass2);
    if (pp1 <= 0) {
        f32 mm = mass1 + std::sqrt(mass2*mass2 + pp2);
        f32 m2 = mm*mm - pp2;
        return m2 >= 0 ? std::sqrt(m2) : std::sqrt(-m2);
    }
    if (pp2 <= 0) {
        f32 mm = mass2 + std::sqrt(mass1*mass1 + pp1);
        f32 m2 = mm*mm - pp1;
        return m2 >= 0 ? std::sqrt(m2) : std::sqrt(-m2);
    }

    const f32 mm1 =  mass1 * mass1;
    const f32 mm2 =  mass2 * mass2;

    const f32 r1 = mm1 / pp1;
    const f32 r2 = mm2 / pp2;
    const f32 x = r1 + r2 + r1 * r2;
    const f32 a = angle(x1, y1, z1, x2, y2, z2);
    const f32 cos_a = std::cos(a);
    f32 y;
    if (cos_a >= 0){
        y = (x + std::sin(a) * std::sin(a)) / (std::sqrt(x + 1) + cos_a);
    } else {
        y = std::sqrt(x + 1) - cos_a;
    }

    const f32 z = 2.0f * std::sqrt(pp1 * pp2);

    // Return invariant mass with (+, -, -, -) metric
    return std::sqrt(mm1 + mm2 + y * z);
}

inline f32 invariantMassPxPyPzE(
   const f32 x1, const f32 y1, const f32 z1, const f32 e1,
   const f32 x2, const f32 y2, const f32 z2, const f32 e2
) {
    const f32 pp1 = x1 * x1 + y1 * y1 + z1 * z1;
    const f32 pp2 = x2 * x2 + y2 * y2 + z2 * z2;

    const f32 mm1 = e1 * e1 - pp1;
    const f32 mm2 = e2 * e2 - pp2;

    const f32 mass1 = (mm1 >= 0) ? std::sqrt(mm1) : 0;
    const f32 mass2 = (mm2 >= 0) ? std::sqrt(mm2) : 0;

    return invariantMassPxPyPzM(x1, y1, z1, mass1, x2, y2, z2, mass2);
}

inline f32 invariantMassPtEtaPhiE(
    const f32 pt1, const f32 eta1, const f32 phi1, const f32 e1,
    const f32 pt2, const f32 eta2, const f32 phi2, const f32 e2
) {
    const f32 x1 = pt1 * std::cos(phi1);
    const f32 y1 = pt1 * std::sin(phi1);
    const f32 z1 = pt1 * std::sinh(eta1);

    const f32 x2 = pt2 * std::cos(phi2);
    const f32 y2 = pt2 * std::sin(phi2);
    const f32 z2 = pt2 * std::sinh(eta2);

    return invariantMassPxPyPzE(x1, y1, z1, e1, x2, y2, z2, e2);
}

inline f32 forwardFolding(
    const f32 recoPt,
    const f32 truePt,
    const f32 s,
    const f32 r
) {
    return s * recoPt + (recoPt - truePt) * (r - s);
}

f32 foldedMass(
    f32 recoPt1, const f32 recoEta1, const f32 recoPhi1, const f32 recoE1,
    f32 recoPt2, const f32 recoEta2, const f32 recoPhi2, const f32 recoE2,
    const f32 truePt1, const f32 truePt2,
    const f32 scale, const f32 resolution
) {
    // Apply forward folding if both truePt values are valid.
    if (truePt1 >= 0 && truePt2 >= 0) {
        recoPt1 = forwardFolding(recoPt1, truePt1, scale, resolution);
        recoPt2 = forwardFolding(recoPt2, truePt2, scale, resolution);
    }

    // Return Invariant mass of sum.
    return invariantMassPtEtaPhiE(
        recoPt1, recoEta1, recoPhi1, recoE1,
        recoPt2, recoEta2, recoPhi2, recoE2
    ) / 1e3f;
}
