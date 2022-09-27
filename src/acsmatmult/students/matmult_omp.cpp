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

void transposeMatrixFloat(float *src, float *dst, uint32_t rows, uint32_t columns) {
#pragma omp for
  for(uint32_t i = 0; i < (rows * columns); i++){
    uint32_t j = i/rows;
    uint32_t k = i%rows;
    dst[i] = src[columns*k + j];
  }
  return;
}

void transposeMatrixDouble(double *src, double *dst, uint32_t rows, uint32_t columns) {
  #pragma omp for
  for(uint32_t i = 0; i < (rows * columns); i++){
    uint32_t j = i/rows;
    uint32_t k = i%rows;
    dst[i] = src[columns*k +j];
  }
  return;
}

// Matrix<float> multiplyMatricesOMP(Matrix<float> a,
//                                   Matrix<float> b,
//                                   int num_threads) {
      
//   /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
//   /* YOU MUST USE OPENMP HERE */
//   auto result = Matrix<float>(a.rows, b.columns);
//   auto matrix_BT = Matrix<float>(b.rows, b.columns);

//   transposeMatrixFloat(&b[0], &matrix_BT[0], (uint32_t)b.rows, (uint32_t)b.columns);

//   float *A = &a[0];
//   float *B = &matrix_BT[0];
//   float *C = &result[0];

//   uint32_t i, j, k;

//   // printf("Matrix A: ");
//   // a.print();
//   // printf("Matrix B: ");
//   // matrix_BT.print();
//   int full_count = matrix_BT.rows/4;
//   int rest_count = matrix_BT.rows%4;
//   omp_set_num_threads(num_threads);
//   #pragma omp for private(i, j, k)
//   for(i = 0; i < a.rows; i++){
//     for(j = 0; j < matrix_BT.rows; j++){
//       float res = 0;
//       avx prod;
//       for(k = 0; k < (uint32_t)full_count * 4; k += 4){
//         auto vec_mat1 = _mm256_loadu_ps(&A[i*a.rows+k]); //load 8 floats
//         // for(int t =0; t < 4; t++){
//         //   printf("vec_mat1 %f\n", vec_mat1.arr[t]);
//         // }
//         // printf("\n");
//         auto vec_mat2 = _mm256_loadu_ps(&B[j*b.rows+k]); //load 8 floats
//         // for(int t =0; t < 4; t++){
//         //   printf("vec_mat2 %f\n", vec_mat2.arr[t]);
//         // }
//         // printf("\n");
//         prod.val = _mm256_mul_ps(vec_mat1, vec_mat2); //dot product
//         // for(int t =0; t < 4; t++){
//         //   printf("prod %f\n", prod.arr[t]);
//         // }
//         // printf("\n");
//         // printf("sum: %f %f %f %f\n", sum.arr[0], sum.arr[1], sum.arr[2], sum.arr[3]);
//         // printf("\n");
//         /*Horizontal add*/
//         res += (prod.arr[0] + prod.arr[1] + prod.arr[2] + prod.arr[3]);
//       }

//       if(rest_count){
//         avx prod2;
//         __m256 rest_a, rest_b;
//         rest_a = _mm256_setzero_ps();
//         rest_b = _mm256_setzero_ps();
//         prod2.val = _mm256_setzero_ps();
//         if(rest_count == 1){
//           __m256i mask = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0xffffffffffffffff);
//           rest_a = _mm256_maskload_ps(&A[i*a.rows+k], mask);
//           rest_b = _mm256_maskload_ps(&B[j*b.rows+k], mask);
//           prod2.val = _mm256_mul_ps(rest_a, rest_b); //dot product
//           res += prod2.arr[0];
//           // printf("REST1 val: %f\n", rest_a.arr[0]);
//           // printf("REST2 val: %f\n", rest_b.arr[0]);
//           // printf("SUM val: %f\n", prod2.arr[0]);
//         }
//         else if(rest_count == 2){
//           __m256i mask = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0xffffffffffffffff, 0xffffffffffffffff);
//           rest_a = _mm256_maskload_ps(&A[i*a.rows+k], mask);
//           rest_b = _mm256_maskload_ps(&B[j*b.rows+k], mask);
//           prod2.val = _mm256_mul_ps(rest_a, rest_b); //dot product
//           res += (prod2.arr[0] + prod2.arr[1]);
//           // printf("REST1 val: %f, %f\n", rest_a.arr[0], rest_a.arr[1]);
//           // printf("REST2 val: %f, %f\n", rest_b.arr[0], rest_b.arr[1]);
//           // printf("SUM val: %f, %f\n", prod2.arr[0], prod2.arr[1]);
//         }
//         else if(rest_count == 3){
//           __m256i mask = _mm256_set_epi64x(0x0000000000000000, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff);
//           rest_a = _mm256_maskload_ps(&A[i*a.rows+k], mask);
//           rest_b = _mm256_maskload_ps(&B[j*b.rows+k], mask);
//           prod2.val = _mm256_mul_ps(rest_a, rest_b); //dot product
//           res+= (prod2.arr[0] + prod2.arr[1] + prod2.arr[2]);
//           // printf("REST1 val: %f, %f, %f\n", rest_a.arr[0], rest_a.arr[1], rest_a.arr[2]);
//           // printf("REST2 val: %f, %f, %f\n", rest_b.arr[0], rest_b.arr[1], rest_b.arr[2]);
//           // printf("SUM val: %f, %f, %f\n", prod2.arr[0], prod2.arr[1], prod2.arr[2]);

//         }

//       }
//       full_count = matrix_BT.rows/4;
//       //Row and row multiplication done
//       //printf("Value: %f\n", res);
//       C[i*a.rows +j] = res;
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
//   auto matrix_BT = Matrix<double>(b.rows, b.columns);

//   transposeMatrixDouble(&b[0], &matrix_BT[0], (uint32_t)b.rows, (uint32_t)b.columns);

//   alignas(64) double *A = &a[0];
//   alignas(64) double *B = &matrix_BT[0];
//   alignas(64) double *C = &result[0];

//   uint32_t i, j, k;

//   // printf("Matrix A: ");
//   // a.print();
//   // printf("Matrix B: ");
//   // matrix_BT.print();

//   int full_count = matrix_BT.rows/4;
//   int rest_count = matrix_BT.rows%4;
//   omp_set_num_threads(num_threads);
//   #pragma omp for private(i, j, k)
//   for(i = 0; i < a.rows; i++){
//     for(j = 0; j < matrix_BT.rows; j++){
//       double res = 0;
//       avxd prod;
//       for(k = 0; k < (uint32_t)full_count * 4; k += 4){
//         auto vec_mat1 = _mm256_loadu_pd(&A[i*a.rows+k]); //load 4 doubles
//         // for(int t =0; t < 4; t++){
//         //   printf("vec_mat1 %f\n", vec_mat1.arr[t]);
//         // }
//         // printf("\n");
//         auto vec_mat2 = _mm256_loadu_pd(&B[j*b.rows+k]); //load 4 doubles
//         // for(int t =0; t < 4; t++){
//         //   printf("vec_mat2 %f\n", vec_mat2.arr[t]);
//         // }
//         // printf("\n");
//         prod.val = _mm256_mul_pd(vec_mat1, vec_mat2); //dot product
//         // for(int t =0; t < 4; t++){
//         //   printf("prod %f\n", prod.arr[t]);
//         // }
//         // printf("\n");
//         // printf("sum: %f %f %f %f\n", sum.arr[0], sum.arr[1], sum.arr[2], sum.arr[3]);
//         // printf("\n");
//         /*Horizontal add*/
//         res += (prod.arr[0] + prod.arr[1] + prod.arr[2] + prod.arr[3]);
//       }

//       if(rest_count){
//         avxd prod2;
//         __m256d rest_a, rest_b;
//         rest_a = _mm256_setzero_pd();
//         rest_b = _mm256_setzero_pd();
//         prod2.val = _mm256_setzero_pd();
//         if(rest_count == 1){
//           __m256i mask = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0xffffffffffffffff);
//           rest_a = _mm256_maskload_pd(&A[i*a.rows+k], mask);
//           rest_b = _mm256_maskload_pd(&B[j*b.rows+k], mask);
//           prod2.val = _mm256_mul_pd(rest_a, rest_b); //dot product
//           res += prod2.arr[0];
//           // printf("REST1 val: %f\n", rest_a.arr[0]);
//           // printf("REST2 val: %f\n", rest_b.arr[0]);
//           // printf("SUM val: %f\n", prod2.arr[0]);
//         }
//         else if(rest_count == 2){
//           __m256i mask = _mm256_set_epi64x(0x0000000000000000, 0x0000000000000000, 0xffffffffffffffff, 0xffffffffffffffff);
//           rest_a = _mm256_maskload_pd(&A[i*a.rows+k], mask);
//           rest_b = _mm256_maskload_pd(&B[j*b.rows+k], mask);
//           prod2.val = _mm256_mul_pd(rest_a, rest_b); //dot product
//           res += (prod2.arr[0] + prod2.arr[1]);
//           // printf("REST1 val: %f, %f\n", rest_a.arr[0], rest_a.arr[1]);
//           // printf("REST2 val: %f, %f\n", rest_b.arr[0], rest_b.arr[1]);
//           // printf("SUM val: %f, %f\n", prod2.arr[0], prod2.arr[1]);
//         }
//         else if(rest_count == 3){
//           __m256i mask = _mm256_set_epi64x(0x0000000000000000, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff);
//           rest_a = _mm256_maskload_pd(&A[i*a.rows+k], mask);
//           rest_b = _mm256_maskload_pd(&B[j*b.rows+k], mask);
//           prod2.val = _mm256_mul_pd(rest_a, rest_b); //dot product
//           res+= (prod2.arr[0] + prod2.arr[1] + prod2.arr[2]);
//           // printf("REST1 val: %f, %f, %f\n", rest_a.arr[0], rest_a.arr[1], rest_a.arr[2]);
//           // printf("REST2 val: %f, %f, %f\n", rest_b.arr[0], rest_b.arr[1], rest_b.arr[2]);
//           // printf("SUM val: %f, %f, %f\n", prod2.arr[0], prod2.arr[1], prod2.arr[2]);

//         }
//         // prod.val = _mm256_mul_pd(vec_mat1.val, vec_mat2.val); //dot product
//         // sum.val = _mm256_add_pd(prod.val, sum.val); //addition
//         //           _mm256_set_pd();
//       }
//       full_count = matrix_BT.rows/4;
//       //Row and row multiplication done
//       //printf("Value: %f\n", res);
//       C[i*a.rows +j] = res;
//     }
//   }
//   // }
//   return result;
// }
//   #pragma GCC pop_options

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


