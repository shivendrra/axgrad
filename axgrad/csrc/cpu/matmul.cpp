#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <stddef.h>
#include <immintrin.h>
#include "matmul.h"

#define HYBRID_BLOCK_SIZE 64

void transpose_2d_array_ops(float* a, float* out, int* shape) {
  int rows = shape[0], cols = shape[1];
  // Transpose: out[j][i] = a[i][j]
  for (int idx = 0; idx < rows * cols ; ++idx) {
    // out is cols x rows, so out[j][i] = out[j * rows + i]
    // a is rows x cols, so a[i][j] = a[idx]
    int i = idx / cols, j = idx % cols;
    out[j * rows + i] = a[idx];
    // this brings down time complexity from O(n2) -> O(n)
  }
}

// Optimized matrix multiplication using transposed second matrix
// A: shape_a[0] x shape_a[1], B^T: shape_b[1] x shape_b[0], C: shape_a[0] x shape_b[0]
// This computes C = A @ B where B is provided in transposed form
void transposed_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int rows_a = shape_a[0];    // rows in 'a'
  int cols_a = shape_a[1];    // cols in 'a' 
  int rows_b = shape_b[0];    // rows in 'b' (original 'b' before transpose)
  int cols_b = shape_b[1];    // cols in 'b' (original 'b' before transpose)
  float* b_transposed = (float*)malloc(rows_b * cols_b * sizeof(float));
  if (b_transposed == NULL) {
    fprintf(stderr, "Memory allocation failed for transpose buffer\n");
    exit(EXIT_FAILURE);
  }
  transpose_2d_array_ops(b, b_transposed, shape_b);
  // for a @ b^T: a(rows_a × cols_a) @ b^T(cols_b × rows_b) = out(rows_a × rows_b)
  // we need cols_a == cols_b for this to work
  for (int i = 0; i < rows_a; i++) {
    for (int j = 0; j < cols_b; j++) {
      float sum = 0.0f;
      // dot product between row i of A and row j of B^T (which is column j of original B)
      for (int k = 0; k < cols_a; k++) {
        sum += a[i * cols_a + k] * b_transposed[j * cols_a + k];
      }
      out[i * cols_b + j] = sum;
    }
  }
}

// void hybrid_transpose_2d_array_ops(float* a, float* out, int* shape) { uncomment this as well for normal test cases
void hybrid_transpose_2d_array_ops(float* a, float* out, int* shape, int block_size) {
  int rows = shape[0], cols = shape[1];
  #pragma omp parallel for schedule(static)
  for (int ii = 0; ii < rows; ii += block_size) {
    for (int jj = 0; jj < cols; jj += block_size) {
      int i_end = (ii + block_size < rows) ? ii + block_size : rows;
      int j_end = (jj + block_size < cols) ? jj + block_size : cols;
      for (int i = ii; i < i_end; i++) {
        for (int j = jj; j < j_end; j++) { out[j * rows + i] = a[i * cols + j]; }
      }
    }
  }
}

void hybrid_transposed_matmul_impl(float* a, float* b, float* out, int* shape_a, int* shape_b, int block_size) {
  int M = shape_a[0], K = shape_a[1], N = shape_b[1];
  int K8 = K & ~7;

  float* bt = aligned_malloc_32(K * N * sizeof(float));
  for (int i = 0; i < shape_b[0]; i++)
    for (int j = 0; j < shape_b[1]; j++)
      bt[j * K + i] = b[i * N + j];

  memset(out, 0, M * N * sizeof(float));

  #pragma omp parallel for collapse(2) schedule(static)
  for (int ii = 0; ii < M; ii += block_size) {
    for (int jj = 0; jj < N; jj += block_size) {

      int ie = ii + block_size < M ? ii + block_size : M;
      int je = jj + block_size < N ? jj + block_size : N;

      for (int i = ii; i < ie; i++) {
        for (int j = jj; j < je; j++) {
          __m256 acc = _mm256_setzero_ps();
          int k = 0;

          for (; k < K8; k += 8) {
            __m256 av = _mm256_loadu_ps(&a[i * K + k]);
            __m256 bv = _mm256_loadu_ps(&bt[j * K + k]);
            acc = _mm256_fmadd_ps(av, bv, acc);
          }

          __m128 hi = _mm256_extractf128_ps(acc, 1);
          __m128 lo = _mm256_castps256_ps128(acc);
          __m128 sum = _mm_add_ps(hi, lo);
          sum = _mm_hadd_ps(sum, sum);
          sum = _mm_hadd_ps(sum, sum);

          float s = _mm_cvtss_f32(sum);
          for (; k < K; k++) s += a[i * K + k] * bt[j * K + k];

          out[i * N + j] = s;
        }
      }
    }
  }
  aligned_free(bt);
}

void hybrid_transposed_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  hybrid_transposed_matmul_impl(a, b, out, shape_a, shape_b, HYBRID_BLOCK_SIZE);
}