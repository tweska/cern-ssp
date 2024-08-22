#ifndef UTIL_H
#define UTIL_H

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
b8 checkArray(const usize n, const T *observed, const T *expected, const f64 maxError = 0.000000000001, const isize maxReports = 25, std::ostream &output = std::cerr) {
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
void printArray(T* array, usize n, std::ostream &output = std::cout)
{
    output << "[ ";
    for (usize i = 0; i < n; ++i) {
        output << array[i] << " ";
    }
    output << "]" << std::endl;
}

#endif //UTIL_H
