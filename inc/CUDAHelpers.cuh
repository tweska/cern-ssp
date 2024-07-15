/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <string>
#include <iostream>
#include <thrust/binary_search.h>

#define ERRCHECK(err) __checkCudaErrors((err), __func__, __FILE__, __LINE__)
inline static void __checkCudaErrors(cudaError_t error, std::string func, std::string file, int line)
{
   if (error != cudaSuccess) {
      std::cerr << func << "(), " << file << ":" << std::to_string(line) << ", " << cudaGetErrorString(error) << std::endl;
      exit(-1);
   }
}

namespace ROOT {
namespace Experimental {
namespace CUDAHelpers {

// Dynamic shared memory needs to be declared as "extern" in CUDA. Having templated kernels with shared memory
// of different data types results in a redeclaration error if the name of the array is the same, so we use a
// proxy function to initialize shared memory arrays of different types with different names.

template <typename T>
inline __device__ T *shared_memory_proxy()
{
   // Fatal("template <typename T> __device__ T *shared_memory_proxy()", "Unsupported shared memory type");
   return (T *)0;
};
template <>
inline __device__ double *shared_memory_proxy<double>()
{
   extern __shared__ double s_double[];
   return s_double;
}

////////////////////////////////////////////////////////////////////////////////
/// CUDA Kernels

// CUDA version of TMath::BinarySearch
template <typename T>
inline __device__ long BinarySearch(long n, const T *array, T value)
{
   const T *pind;

   pind = thrust::lower_bound(thrust::seq, array, array + n, value);

   if ((pind != array + n) && (*pind == value))
      return (pind - array);
   else
      return (pind - array - 1);

   // return pind - array - !((pind != array + n) && (*pind == value)); // OPTIMIZATION: is this better?
}

// For debugging
inline __global__ void PrintArray(double *array, int n)
{
   for (int i = 0; i < n; i++) {
      printf("%f ", array[i]);
   }
   printf("\n");
}

} // namespace CUDAHelpers
} // namespace Experimental
} // namespace ROOT
#endif
