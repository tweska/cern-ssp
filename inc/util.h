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
void printArray(T* array, usize n, std::ostream &output = std::cout)
{
    output << "[ ";
    for (usize i = 0; i < n; ++i) {
        output << array[i] << " ";
    }
    output << "]" << std::endl;
}

#endif //UTIL_H
