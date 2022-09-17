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
  double arr[128];
} avxd;

typedef union _avxs {
  __m256 val;
  float arr[128];
} avxs;

Matrix<float> multiplyMatricesSIMD(Matrix<float> a, Matrix<float> b) {
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE VECTOR EXTENSIONS HERE */
  avxs vec_a, vec_b, res;

  if (a.rows != b.columns){
    return Matrix<float>(1, 1);
  }

  auto result = Matrix<float>(a.rows, b.columns);

  float ind;
  if (a.rows == 1 && a.columns == 1){
    vec_a.val = _mm256_set1_ps(a.operator()(0, 0)); // Matrix A stored as vector
    vec_b.val = _mm256_set1_ps(b.operator()(0, 0)); //Inserts the column of matrix b
    res.val = _mm256_mul_ps((__m256)vec_a.val, (__m256)vec_b.val);
    _mm256_storeu_ps((float *)&ind, res.val);
    result(0, 0) = ind;
    return result;
  }

  for(uint16_t i = 0; i < (uint16_t)a.rows; i++){
    for(uint16_t j= 0; j < (uint16_t)b.columns; j++){
      res.val = _mm256_setzero_ps(); //Set the vector to 0
      for(uint16_t k = 0; k < (uint16_t)b.rows; k++){
        vec_a.val = _mm256_set1_ps(a.operator()(i, k)); // Matrix A stored as vector
        vec_b.val = _mm256_set1_ps(b.operator()(k, j)); //Inserts the column of matrix b
        res.val = _mm256_add_ps((__m256)res.val, _mm256_mul_ps((__m256)vec_a.val, (__m256)vec_b.val)); //Finally.. multiplication of the vectors and add the result
        _mm256_storeu_ps((float *)&result(i, j), (__m256)res.val);
      }
    }
  }
  return result;
}

Matrix<double> multiplyMatricesSIMD(Matrix<double> a,
                                  Matrix<double> b) {
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE VECTOR EXTENSIONS HERE */
  auto vec_a = _mm256_setzero_pd(); //Set the vector to 0
  auto vec_b = _mm256_setzero_pd(); //Set the vector to 0
  auto vec_matrix_res = _mm256_setzero_pd(); //Set the vector to 0

  if (a.rows != b.columns){
    return Matrix<double>(1, 1);
  }

  auto result = Matrix<double>(a.rows, b.columns);

  double res;
  if (a.rows == 1 && a.columns == 1){
    vec_a = _mm256_set1_pd(a.operator()(0, 0)); // Matrix A stored as vector
    vec_b = _mm256_set1_pd(b.operator()(0, 0)); //Inserts the column of matrix b
    vec_matrix_res = _mm256_mul_pd(vec_a, vec_b);
    _mm256_storeu_pd((double *)&res, vec_matrix_res);
    result(0, 0) = res;
    return result;
  }

  for(uint16_t i = 0; i < (uint16_t)a.rows; i++){
    for(uint16_t j= 0; j < (uint16_t)b.columns; j++){
      vec_matrix_res = _mm256_setzero_pd(); //Set the vector to 0
      for(uint16_t k = 0; k < (uint16_t)b.rows; k++){
        vec_a = _mm256_set1_pd(a.operator()(i, k)); // Matrix A stored as vector
        vec_b = _mm256_set1_pd(a.operator()(k, j)); //Inserts the column of matrix b
        vec_matrix_res = _mm256_add_pd(vec_matrix_res, _mm256_mul_pd(vec_a, vec_b)); //Finally.. multiplication of the vectors and add the result
        _mm256_storeu_pd((double *)&res, vec_matrix_res);
        result(i, j) = res;
      }
    }
  }
  return result;
}

/*************************************/
#pragma GCC pop_options
/*************************************/
