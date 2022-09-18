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
#include <immintrin.h>  // Intel intrinsics for SSE/AVX.

/* You may not remove these pragmas: */
/*************************************/
#pragma GCC push_options
#pragma GCC optimize ("O0")
/*************************************/

typedef union _avxd {
  __m256d val;
  double arr[4];
} avxd;

typedef union _avxs {
  __m256 val;
   float arr[4];
} avxs;

Matrix<float> multiplyMatricesSIMD(Matrix<float> a, Matrix<float> b) {
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE VECTOR EXTENSIONS HERE */
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE VECTOR EXTENSIONS HERE */
  auto result = Matrix<float>(a.rows, b.columns);
  alignas(32) float num;
  
  avxs res;

  for(uint16_t i = 0; i < (uint16_t)a.rows; i++){
    for(uint16_t j= 0; j < (uint16_t)b.columns; j++){
      res.val = _mm256_setzero_ps(); //Set the vector to 0
      for(uint16_t k = 0; k < (uint16_t)b.rows; k++){
        __m256 vec_a = _mm256_set1_ps(a.operator()(i, k)); // Matrix A stored as vector
        // printf("A[%d][%d]: %f x " , i, k, a.operator()(i, k));
        __m256 vec_b = _mm256_set1_ps(b.operator()(k, j)); //Inserts the column of matrix b
        // printf("B[%d][%d]: %f = " , k, j, b.operator()(k, j));
        res.val = _mm256_add_ps(res.val, _mm256_mul_ps(vec_a, vec_b));
        _mm256_store_ps((float *)&num, res.val);
        // printf("R[%d][%d]: %f\n", i, j, num);
        result(i, j) = num;
      }
    }
  }
  return result;
}  

Matrix<double> multiplyMatricesSIMD(Matrix<double> a,
                                  Matrix<double> b) {
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE VECTOR EXTENSIONS HERE */
  auto result = Matrix<double>(a.rows, b.columns);
  alignas(64) double num;

  avxd res;

  for(uint16_t i = 0; i < (uint16_t)a.rows; i++){
    for(uint16_t j= 0; j < (uint16_t)b.columns; j++){
      res.val = _mm256_setzero_pd(); //Set the vector to 0
      for(uint16_t k = 0; k < (uint16_t)b.rows; k++){
        __m256d vec_a = _mm256_set1_pd(a.operator()(i, k)); // Matrix A stored as vector
        // printf("A[%d][%d]: %f x " , i, k, a.operator()(i, k));
        __m256d vec_b = _mm256_set1_pd(b.operator()(k, j)); //Inserts the column of matrix b
        // printf("B[%d][%d]: %f = " , k, j, b.operator()(k, j));
        res.val = _mm256_add_pd(res.val, _mm256_mul_pd(vec_a, vec_b));
        _mm256_store_pd((double *)&num, res.val);
        // printf("R[%d][%d]: %f\n", i, j, num);
        result(i, j) = num;
      }
    }
  }
  return result;
}

/*************************************/
#pragma GCC pop_options
/*************************************/
