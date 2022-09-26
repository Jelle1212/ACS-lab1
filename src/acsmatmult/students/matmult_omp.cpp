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

 float * A = (float *) aligned_alloc(n, sizeof(float) * n);
 float * B = (float *) aligned_alloc(n, sizeof(float) * n);
 float * C = (float *) aligned_alloc(n, sizeof(float) * n);
  
  A = &a[0];
  B = &b[0];
  C = &result[0];



  uint32_t i, j, k;
  omp_set_num_threads(num_threads);
  #pragma omp parallel shared(C) private(i,j,k)
  {
    for(i = 0; i < n; i++){
      for(j = 0; j < n; j++){
        auto sum = _mm256_setzero_ps();
        for(k = 0; k < n; k += 8){
          auto bc_mat1 = _mm256_set1_ps(*A+(i*n)+k);
          auto vec_mat2 = _mm256_load_ps(B+(j*n)+k);
          auto prod = _mm256_mul_ps(bc_mat1, vec_mat2);
          sum = _mm256_add_ps(sum, prod);
        }
        //_mm256_storeu_si256((__m256i*)&C[i*n + j], sum); 
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
    for(i = 0; i < n; i++){
      for(j = 0; j < n; j++){
        auto sum = _mm256_setzero_pd();
        for(k = 0; k < n; k += n){
          auto bc_mat1 = _mm256_set1_pd(*A+(i*n)+k);
          auto vec_mat2 = _mm256_load_pd(B+(j*n)+k);
          auto prod = _mm256_mul_pd(bc_mat1, vec_mat2);
          sum = _mm256_add_pd(sum, prod);
        }
        _mm256_store_pd(&C[i*n + j], sum); 
      }
    }
  }
  return result;
}
  #pragma GCC pop_options


