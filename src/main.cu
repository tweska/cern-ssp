#include <iostream>

#include "RHnCUDA.h"

using namespace ROOT::Experimental;

int main(int argc, char **argv)
{
    HistogramGlobal<double, 1><<<1, 1>>>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1);
    cudaDeviceSynchronize();
    ERRCHECK(cudaPeekAtLastError());  // This should crash!

    std::cout << "Hello, World!" << std::endl;

    return 0;
}
