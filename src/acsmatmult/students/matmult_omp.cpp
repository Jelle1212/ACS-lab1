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

/* You may not remove these pragmas: */
/*************************************/
#pragma GCC push_options
#pragma GCC optimize ("O0")
/*************************************/

// void transposeMatrixFloat(Matrix<float> src, Matrix<float> dst) {
//   uint32_t row = src.rows;
//   uint32_t col = src.columns;
//   uint32_t n =  row * col;

//   for(uint32_t i = 0; i < n; i++){
//     uint32_t j = n/row;
//     uint32_t k = n%row;
//     dst[n] = src[col*k +j];
//   }
//   return;
// }

// void transposeMatrixDouble(Matrix<double> src, Matrix<double> dst) {
//   uint32_t row = src.rows;
//   uint32_t col = src.columns;
//   uint32_t n =  row * col;


//   for(uint32_t i = 0; i < n; i++){
//     uint32_t j = n/row;
//     uint32_t k = n%row;
//     printf("test: %f", src[col*k +j]);
//     dst[n] = src[col*k +j];
//   }
//   return;
// }

// Matrix<float> multiplyMatricesOMP(Matrix<float> a,
//                                   Matrix<float> b,
//                                   int num_threads) {
      
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE OPENMP HERE */
  // uint32_t N = a.rows;
  // auto result = Matrix<float>(N, N);
  // auto b_tran = Matrix<float>(N, N);

  // for(uint32_t n = 0; n < (N*N); n++) {
  //     uint32_t i = n/N;
  //     uint32_t j = n%N;
  //     b_tran[n] = b[N*j + i];
  // }

  // printf("A: ");
  // a.print();
  // printf("B: ");
  // b.print();
  // printf("B^t: ");
  // b_tran.print();
//   uint32_t i, j, k;
//   omp_set_num_threads(num_threads);
//   #pragma omp parallel for shared(result) private(i, j, k)
//   for(i = 0; i < N; i++){
//     for(j = 0; j < N; j++){
//       float val = 0;
//       for(k = 0; k <N; k++){
//         val += a[i * N + k] * b[k * N + j];
//       }
//       result[i *N + j] = val;
//     }
//   }
//   return result;
// }

Matrix<float> multiplyMatricesOMP(Matrix<float> a,
                                  Matrix<float> b,
                                  int num_threads) {
      
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE OPENMP HERE */
  uint32_t n = a.rows;
  auto result = Matrix<float>(n, n);
  
  uint32_t i, j, k;
  omp_set_num_threads(num_threads);
  #pragma omp parallel for shared(result) private(i, j, k)
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      for(k = 0; k <n; k++){
        result(i, j) += a(i, k) * b(k, j);
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
  auto result = Matrix<double>(a.rows, b.columns);

  uint32_t i, j, k;
  omp_set_num_threads(num_threads);
  #pragma omp parallel for shared(result) private(i, j, k)
    for(i = 0; i < a.rows; i++){
      for(j = 0; j < b.rows; j++){
        for(k = 0; k <b.columns; k++){
          result(i, j) += a(i, k) * b(k, j);
        }
      }
    }
    return result;
  }
#pragma GCC pop_options
