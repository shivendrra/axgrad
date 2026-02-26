#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <stddef.h>
#include <immintrin.h>
#include "matmul.h"

#pragma GCC target("avx2,fma")
#pragma GCC optimize("O3")

#define HYBRID_BLOCK_SIZE 64

void transpose_2d_array_ops(float* a, float* out, int* shape) {
  int rows = shape[0], cols = shape[1];
  for (int idx = 0; idx < rows * cols; ++idx) {
    int i = idx / cols, j = idx % cols;
    out[j * rows + i] = a[idx];
  }
}

void hybrid_transpose_2d_array_ops(float* a, float* out, int* shape, int block_size) {
  int rows = shape[0], cols = shape[1];
  #pragma omp parallel for schedule(static)
  for (int ii = 0; ii < rows; ii += block_size) {
    for (int jj = 0; jj < cols; jj += block_size) {
      int i_end = (ii + block_size < rows) ? ii + block_size : rows;
      int j_end = (jj + block_size < cols) ? jj + block_size : cols;
      for (int i = ii; i < i_end; i++)
        for (int j = jj; j < j_end; j++) out[j * rows + i] = a[i * cols + j];
    }
  }
}

void transposed_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int rows_a = shape_a[0], cols_a = shape_a[1];
  int rows_b = shape_b[0], cols_b = shape_b[1];
  float* bt = (float*)malloc(rows_b * cols_b * sizeof(float));
  if (!bt) { fprintf(stderr, "Memory allocation failed\n"); exit(EXIT_FAILURE); }
  transpose_2d_array_ops(b, bt, shape_b);
  for (int i = 0; i < rows_a; i++) {
    for (int j = 0; j < cols_b; j++) {
      float sum = 0.0f;
      for (int k = 0; k < cols_a; k++) sum += a[i * cols_a + k] * bt[j * cols_a + k];
      out[i * cols_b + j] = sum;
    }
  }
  free(bt);
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
      int ie = (ii + block_size < M) ? ii + block_size : M;
      int je = (jj + block_size < N) ? jj + block_size : N;
      for (int i = ii; i < ie; i++) {
        for (int j = jj; j < je; j++) {
          __m256 acc = _mm256_setzero_ps();
          int k = 0;
          for (; k < K8; k += 8) {
            __m256 av = _mm256_loadu_ps(&a[i * K + k]);
            __m256 bv = _mm256_loadu_ps(&bt[j * K + k]);
            acc = _mm256_fmadd_ps(av, bv, acc);
          }
          __m128 lo = _mm256_castps256_ps128(acc);
          __m128 hi = _mm256_extractf128_ps(acc, 1);
          __m128 s  = _mm_add_ps(lo, hi);
          s = _mm_hadd_ps(s, s);
          s = _mm_hadd_ps(s, s);
          float sum = _mm_cvtss_f32(s);
          for (; k < K; k++) sum += a[i * K + k] * bt[j * K + k];
          out[i * N + j] = sum;
        }
      }
    }
  }
  aligned_free(bt);
}

void hybrid_transposed_matmul(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  hybrid_transposed_matmul_impl(a, b, out, shape_a, shape_b, HYBRID_BLOCK_SIZE);
}