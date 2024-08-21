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
b8 checkArray(const usize n, const T *observed, const T *expected, const f64 maxError = 0.000000000001) {
    for (usize i = 0; i < n; ++i)
        if (fabsl((observed[i] - expected[i]) / expected[i]) > maxError)
            return false;
    return true;
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
