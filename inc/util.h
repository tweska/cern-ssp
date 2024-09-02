#ifndef UTIL_H
#define UTIL_H

#include <cassert>

#include <iomanip>
#include <iostream>
#include <numeric>

#include "types.h"

template <typename T>
T arraySum(T a[], size_t n)
{
    return std::accumulate(a, a + n, 0);
}

template <typename T>
T arrayProduct(T a[], size_t n)
{
    return std::accumulate(a, a + n, 1, std::multiplies<T>());
}

template <typename T>
b8 checkArray(
    const usize n, const T *observed, const T *expected,
    const f64 maxError = 0.000000000001, const isize maxReports = 25,
    std::ostream &output = std::cerr
) {
    isize reports = maxReports >= 0 ? 0 : -1;
    usize errors = 0;
    for (usize i = 0; i < n; ++i) {
        const f64 error = fabsl((observed[i] - expected[i]) / expected[i]);
        if (error > maxError) {
            ++errors;
            if (reports >= 0 && reports++ < maxReports) {
                output << "Error at index " << i << " : " << observed[i] << " != " << expected[i] << " (" << error << ")" << std::endl;
            }
        }
    }
    return errors == 0;
}

template <typename T>
b8 checkArrayAvg(
    const usize n, const usize k, const T *observed, const T *expected,
    const f64 maxError = 0.000000000001, const isize maxReports = 25,
    std::ostream &output = std::cerr
) {
    const auto nAvg = n - (k - 1);
    T *observedAvg = new T[nAvg];
    T *expectedAvg = new T[nAvg];
    for (usize i = 0; i < nAvg; ++i) {
        observedAvg[i] = 0.0;
        expectedAvg[i] = 0.0;
        for (usize j = 0; j < k; ++j) {
            observedAvg[i] += observed[i + j];
            expectedAvg[i] += expected[i + j];
        }
    }
    auto result = checkArray(nAvg, observedAvg, expectedAvg, maxError, maxReports, output);
    delete[] observedAvg;
    delete[] expectedAvg;
    return result;
}

template <typename T>
void arrayMaxError(
    T *maxError, T *oValue, T *eValue,
    const usize n, const T *observed, const T *expected,
    const b8 ignoreZero = true
) {
    assert(n > 0);
    *maxError = *oValue = *eValue = 0.0;
    for (usize i = 1; i < n; ++i) {
        if (ignoreZero && expected[i] == 0.0) { continue; }
        const f64 error = fabsl((observed[i] - expected[i]) / expected[i]);
        if (error > *maxError) {
            *maxError = error;
            *oValue = observed[i];
            *eValue = expected[i];
        }
    }
}

template <typename T>
void printArray(T* array, usize n, std::ostream &output = std::cout)
{
    output << "[ ";
    for (usize i = 0; i < n; ++i) {
        output << array[i] << " ";
    }
    output << "]" << std::endl;
}

template <typename T>
void arrayMinMaxAvg(T array[], usize n, T &min, T &max, T &avg)
{
    assert(n >=1);
    T sum = min = max = array[0];

    for (usize i = 1; i < n; ++i) {
        sum += array[i];
        if (array[i] < min)
            min = array[i];
        if (array[i] > max)
            max = array[i];
    }

    avg = sum / n;
}

template <typename T>
void printArrayMinMaxAvg(T array[], usize n, std::string unit = "", const i32 width = 6, const i32 precision = 0, std::ostream &output = std::cerr)
{
    T min, max, avg;
    arrayMinMaxAvg(array, n, min, max, avg);
    output << std::setw(width) << std::fixed << std::setprecision(precision) << min << unit << " (min), "
           << std::setw(width) << std::fixed << std::setprecision(precision) << max << unit << " (max), "
           << std::setw(width) << std::fixed << std::setprecision(precision) << avg << unit << " (avg)" << std::endl;
}

#endif //UTIL_H
