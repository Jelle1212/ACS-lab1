// Copyright 2018 Delft University of Technology
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "acsmatmult/matmult.h"
#include <omp.h>  // OpenMP support.
#include <vector>
#include <immintrin.h>  // Intel intrinsics for SSE/AVX.


/* You may not remove these pragmas: */
/*************************************/
#pragma GCC push_options
#pragma GCC optimize ("O0")
/*************************************/


Matrix<float> multiplyMatricesOMP(Matrix<float> a,
                                  Matrix<float> b,
                                  int num_threads) {
      
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE OPENMP HERE */
  uint32_t n = a.rows;
  auto result = Matrix<float>(n, n);
  
  float *A = &a[0];
  float *B = &b[0];
  float *C = &result[0];

  uint32_t i, j, k;
  omp_set_num_threads(num_threads);
  #pragma omp parallel shared(C) private(i,j,k)
  {
    #pragma omp for schedule(static)
      for(i = 0; i < n; i++){
        for(j = 0; j < n; j++){
          float res = 0;
          for(k = 0; k < n; k++){
            res += A[i*n + k] * B[j*n + k];
          }
          C[i*n + j] = res;
        }
      }
  }
  return result;
}



Matrix<double> multiplyMatricesOMP(Matrix<double> a,
                                   Matrix<double> b,
                                   int num_threads) {
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE OPENMP HERE */
  uint32_t n = a.rows;
  auto result = Matrix<double>(n, n);
  double *A = &a[0];
  double *B = &b[0];
  double *C = &result[0];

  uint32_t i, j, k;
  omp_set_num_threads(num_threads);
  #pragma omp parallel shared(C) private(i,j,k)
  {
    #pragma omp for schedule(static)
      for(i = 0; i < n; i++){
        for(j = 0; j < n; j++){
          float res = 0;
          for(k = 0; k < n; k++){
            res += A[i*n + k] * B[j*n + k];
          }
          C[i*n + j] = res;
        }
      }
  }
  return result;
}
  #pragma GCC pop_options


