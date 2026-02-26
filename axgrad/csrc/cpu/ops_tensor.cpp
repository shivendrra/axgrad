#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include "ops_tensor.h"
#include "ops_shape.h"
#include "matmul.h"

#pragma GCC target("avx2,fma")
#pragma GCC optimize("O3")

// matmul_tensor_ops
// Uses hybrid_transposed_matmul: blocked + AVX2 fmadd + OpenMP — fastest path.

void matmul_tensor_ops(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  hybrid_transposed_matmul(a, b, out, shape_a, shape_b);
}

// batch_matmul_tensor_ops
// Uses transposed_matmul per batch slice.
// A: [batch x M x K],  B: [batch x K x N],  out: [batch x M x N]

void batch_matmul_tensor_ops(float* a, float* b, float* out, int* shape1, int* shape2, int* strides1, int* strides2) {
  int batch = shape1[0];
  int M = shape1[1], K = shape1[2], N = shape2[2];
  int ms_a = M * K, ms_b = K * N, ms_out = M * N;

  int sh_a[2] = {M, K}, sh_b[2] = {K, N};

#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int bi = 0; bi < batch; bi++) {
    float* a_b = a + bi * strides1[0];
    float* b_b = b + bi * strides2[0];  // intentional: b pointer + offset
    float* out_b = out + bi * ms_out;

    // Transpose B slice, then do AVX2 dot rows
    float* bt = aligned_malloc_32(K * N * sizeof(float));
    for (int i = 0; i < K; i++)
      for (int j = 0; j < N; j++)
        bt[j * K + i] = b_b[i * N + j];

    int K8 = K & ~7;
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        __m256 acc = _mm256_setzero_ps();
        int k = 0;
        for (; k < K8; k += 8)
          acc = _mm256_fmadd_ps(_mm256_loadu_ps(a_b + i*K + k), _mm256_loadu_ps(bt  + j*K + k), acc);
        __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc, 1);
        __m128 s = _mm_add_ps(lo, hi); s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
        float sum = _mm_cvtss_f32(s);
        for (; k < K; k++) sum += a_b[i*K + k] * bt[j*K + k];
        out_b[i * N + j] = sum;
      }
    }
    aligned_free(bt);
  }
}

// broadcasted_matmul_tensor_ops
// A is 2D [M x K], broadcast across B's batches [batch x K x N] → out [batch x M x N]

void broadcasted_matmul_tensor_ops(float* a, float* b, float* out, int* shape1, int* shape2, int* strides1, int* strides2) {
  int batch = shape2[0];
  int M = shape1[0], K = shape1[1], N = shape2[2];
  int ms_out = M * N;
  int K8 = K & ~7;

  // Precompute A shape for hybrid call — A is shared across all batches
  int sh_a[2] = {M, K};

#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int bi = 0; bi < batch; bi++) {
    float* b_b = b + bi * strides2[0];
    float* out_b = out + bi * ms_out;

    // Transpose this batch's B slice
    float* bt = aligned_malloc_32(K * N * sizeof(float));
    for (int i = 0; i < K; i++)
      for (int j = 0; j < N; j++)
        bt[j * K + i] = b_b[i * N + j];

    // A is broadcasted — same pointer every batch
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        __m256 acc = _mm256_setzero_ps();
        int k = 0;
        for (; k < K8; k += 8)
          acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i*K + k), _mm256_loadu_ps(bt + j*K + k), acc);
        __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc, 1);
        __m128 s = _mm_add_ps(lo, hi); s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
        float sum = _mm_cvtss_f32(s);
        for (; k < K; k++) sum += a[i*K + k] * bt[j*K + k];
        out_b[i * N + j] = sum;
      }
    }
    aligned_free(bt);
  }
}

// dot_tensor_ops
// AVX2 dot product with OpenMP reduction for large vectors.

void dot_tensor_ops(float* a, float* b, float* out, size_t size) {
  float sum = 0.0f;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static) reduction(+:sum)
  for (size_t i = 0; i < size; i++) sum += a[i] * b[i];
#else
  __m256 vacc = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 8 <= size; i += 8)
    vacc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), vacc);
  __m128 lo = _mm256_castps256_ps128(vacc), hi = _mm256_extractf128_ps(vacc, 1);
  __m128 s = _mm_add_ps(lo, hi); s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
  sum = _mm_cvtss_f32(s);
  for (; i < size; i++) sum += a[i] * b[i];
#endif
  *out = sum;
}

// batch_dot_tensor_ops
// Each batch slice is an independent dot product — parallel over batches,
// AVX2 within each slice.

void batch_dot_tensor_ops(float* a, float* b, float* out, size_t batch_count, size_t vector_size) {
  int vs8 = (int)vector_size & ~7;

#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (size_t batch = 0; batch < batch_count; batch++) {
    size_t off = batch * vector_size;
    __m256 vacc = _mm256_setzero_ps();
    int k = 0;
    for (; k < vs8; k += 8)
      vacc = _mm256_fmadd_ps(_mm256_loadu_ps(a + off + k), _mm256_loadu_ps(b + off + k), vacc);
    __m128 lo = _mm256_castps256_ps128(vacc), hi = _mm256_extractf128_ps(vacc, 1);
    __m128 s = _mm_add_ps(lo, hi); s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    float sum = _mm_cvtss_f32(s);
    for (size_t i = k; i < vector_size; i++) sum += a[off+i] * b[off+i];
    out[batch] = sum;
  }
}