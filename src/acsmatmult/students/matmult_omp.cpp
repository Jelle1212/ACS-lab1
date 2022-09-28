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

typedef union _avxd {
  __m256d val;
  double arr[4];
} avxd;

typedef union _avx {
  __m256 val;
  float arr[4];
} avx;

/* You may not remove these pragmas: */
/*************************************/
#pragma GCC push_options
#pragma GCC optimize ("O0")
/*************************************/

// void transposeMatrixFloat(float *src, float *dst, uint32_t rows, uint32_t columns) {
// #pragma omp for
//   for(uint32_t i = 0; i < (rows * columns); i++){
//     uint32_t j = i/rows;
//     uint32_t k = i%rows;
//     dst[i] = src[columns*k + j];
//   }
//   return;
// }

// void transposeMatrixDouble(double *src, double *dst, uint32_t rows, uint32_t columns) {
//   #pragma omp for
//   for(uint32_t i = 0; i < (rows * columns); i++){
//     uint32_t j = i/rows;
//     uint32_t k = i%rows;
//     dst[i] = src[columns*k +j];
//   }
//   return;
// }

Matrix<float> multiplyMatricesOMP(Matrix<float> a,
                                  Matrix<float> b,
                                  int num_threads) {
      
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE OPENMP HERE */

  auto res = Matrix<float>(a.rows, b.columns);
  float *A = &a[0];
  float *B = &b[0];
  float *C = &res[0];


  uint32_t n = a.rows;
  uint32_t i, j, k;
  #pragma omp parallel for schedule(guided) num_threads(num_threads) shared(A, B, C) private(i, j, k)
    for(i = 0; i < a.rows; i++){
      for(j = 0; j < b.columns; j++){
        float val = 0;
        for(k = 0; k < a.columns; k++){
          val += A[i*n + k]*B[k*n+j];
        }
        C[i*n+j] = val;
      }
    }
  return res;
}



Matrix<double> multiplyMatricesOMP(Matrix<double> a,
                                   Matrix<double> b,
                                   int num_threads) {
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE OPENMP HERE */

  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE OPENMP HERE */

  auto res = Matrix<double>(a.rows, b.columns);
  double *A = &a[0];
  double *B = &b[0];
  double *C = &res[0];


  uint32_t n = a.rows;
  uint32_t i, j, k;
  #pragma omp parallel for schedule(guided) num_threads(num_threads) shared(A, B, C) private(i, j, k)
    for(i = 0; i < a.rows; i++){
      for(j = 0; j < b.columns; j++){
        double val = 0;
        for(k = 0; k < a.columns; k++){
          val += A[i*n + k]*B[k*n+j];
        }
        C[i*n+j] = val;
      }
    }
  return res;
}
#pragma GCC pop_options

// Matrix<float> multiplyMatricesOMP(Matrix<float> a,
//                                   Matrix<float> b,
//                                   int num_threads) {
      
//   /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
//   /* YOU MUST USE OPENMP HERE */
//   uint32_t n = a.rows;
//   auto result = Matrix<float>(n, n);
  
//   uint32_t i, j, k;
//   omp_set_num_threads(num_threads);
//   #pragma omp parallel for
//   for(i = 0; i < n; i++){
//     for(j = 0; j < n; j++){
//       for(k = 0; k <n; k++){
//         result(i, j) += a(i, k) * b(k, j);
//       }
//     }
//   }
//   return result;
// }



// Matrix<double> multiplyMatricesOMP(Matrix<double> a,
//                                    Matrix<double> b,
//                                    int num_threads) {
//   /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
//   /* YOU MUST USE OPENMP HERE */
//   auto result = Matrix<double>(a.rows, b.columns);

//   uint32_t i, j, k;
//   omp_set_num_threads(num_threads);
//   #pragma omp parallel for
//     for(i = 0; i < a.rows; i++){
//       for(j = 0; j < b.rows; j++){
//         for(k = 0; k <b.columns; k++){
//           result(i, j) += a(i, k) * b(k, j);
//         }
//       }
//     }
//     return result;
//   }
// #pragma GCC pop_options


