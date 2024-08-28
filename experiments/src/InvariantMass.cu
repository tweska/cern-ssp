#include "../../inc/types.h"

#include "InvariantMass.h"

#include "GpuDefHisto.cu"

template <typename T>
__device__
T Angle(T x0, T y0, T z0, T x1, T y1, T z1)
{
    // cross product
    const auto cx = y0 * z1 - y1 * z0;
    const auto cy = x0 * z1 - x1 * z0;
    const auto cz = x0 * y1 - x1 * y0;

    // norm of cross product
    const auto c = sqrt(cx * cx + cy * cy + cz * cz);

    // dot product
    const auto  d = x0 * x1 + y0 * y1 + z0 * z1;

    return atan2(c, d);
}

template <typename T>
__device__
T InvariantMassPxPyPzM(
   const T x0, const T y0, const T z0, const T mass0,
   const T x1, const T y1, const T z1, const T mass1
) {
    // Numerically stable computation of Invariant Masses
    const auto p0_sq = x0 * x0 + y0 * y0 + z0 * z0;
    const auto p1_sq = x1 * x1 + y1 * y1 + z1 * z1;

    if (p0_sq <= 0 && p1_sq <= 0)
        return (mass0 + mass1);
    if (p0_sq <= 0) {
        auto mm = mass0 + sqrt(mass1*mass1 + p1_sq);
        auto m2 = mm*mm - p1_sq;
        return m2 >= 0 ? sqrt(m2) : sqrt(-m2);
    }
    if (p1_sq <= 0) {
        auto mm = mass1 + sqrt(mass0*mass0 + p0_sq);
        auto m2 = mm*mm - p0_sq;
        return m2 >= 0 ? sqrt(m2) : sqrt(-m2);
    }

    const auto m0_sq =  mass0 * mass0;
    const auto m1_sq =  mass1 * mass1;

    const auto r0 = m0_sq / p0_sq;
    const auto r1 = m1_sq / p1_sq;
    const auto x = r0 + r1 + r0 * r1;
    const auto a = Angle(x0, y0, z0, x1, y1, z1);
    const auto cos_a = cos(a);
    auto y = x;
    if (cos_a >= 0){
        y = (x + sin(a) * sin(a)) / (sqrt(x + 1) + cos_a);
    } else {
        y = sqrt(x + 1) - cos_a;
    }

    const auto z = 2 * sqrt(p0_sq * p1_sq);

    // Return invariant mass with (+, -, -, -) metric
    return sqrt(m0_sq + m1_sq + y * z);
}


/// @brief Return the invariant mass of two particles given their
/// transverse momentum (pt), rapidity (eta), azimuth (phi) and mass.
template <typename T>
__device__
T inline InvariantMass(
    T pt0, T eta0, T phi0, T mass0,
    T pt1, T eta1, T phi1, T mass1
) {
    const auto x0 = pt0 * cos(phi0);
    const auto y0 = pt0 * sin(phi0);
    const auto z0 = pt0 * sinh(eta0);

    const auto x1 = pt1 * cos(phi1);
    const auto y1 = pt1 * sin(phi1);
    const auto z1 = pt1 * sinh(eta1);

    return InvariantMassPxPyPzM(x0, y0, z0, mass0, x1, y1, z1, mass1);
}

/// @brief Return the invariant mass of two particles given their
/// transverse momentum (pt), rapidity (eta), azimuth (phi) and mass.
__device__
f64 InvariantMass(f64 *coords, usize i, usize n)
{
    const usize offset = i * 8;
    return InvariantMass<f64>(
        coords[offset + 0], coords[offset + 2], coords[offset + 4], coords[offset + 6],
        coords[offset + 1], coords[offset + 3], coords[offset + 5], coords[offset + 7]
    );
}

template class GpuDefHisto<f64, InvariantMass>;
