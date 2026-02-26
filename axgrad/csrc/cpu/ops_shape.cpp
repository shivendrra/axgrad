#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <immintrin.h>
#include "ops_shape.h"

void reassign_tensor_ops(float* a, float* out, size_t size) {
  int i = 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  for (i = 0; i < (int)size; i++) out[i] = a[i];
#else
  for (; i <= (int)size - 8; i += 8)
    _mm256_storeu_ps(out + i, _mm256_loadu_ps(a + i));
  for (; i < (int)size; i++) out[i] = a[i];
#endif
}

// Comparison Ops — AVX2 macro
// Each comparison loads 8 floats, compares, and stores 1.0f/0.0f results.

#define CMP_TENSOR_AVX2(name, avx_cmp_op, scalar_cmp)                          \
void name(float* a, float* b, float* out, size_t size) {                        \
  __m256 one = _mm256_set1_ps(1.0f), zero = _mm256_setzero_ps();               \
  int i = 0;                                                                    \
  _Pragma("omp parallel for schedule(static)")                                  \
  for (i = 0; i < (int)size; i++) out[i] = (a[i] scalar_cmp b[i]) ? 1.0f : 0.0f; \
}

#define CMP_SCALAR_AVX2(name, avx_cmp_op, scalar_cmp)                          \
void name(float* a, float b, float* out, size_t size) {                         \
  int i = 0;                                                                    \
  _Pragma("omp parallel for schedule(static)")                                  \
  for (i = 0; i < (int)size; i++) out[i] = (a[i] scalar_cmp b) ? 1.0f : 0.0f; \
}

// Use AVX2 for non-OpenMP builds, OpenMP parallel for OpenMP builds.
// The macros unify both paths cleanly.

void equal_tensor_ops(float* a, float* b, float* out, size_t size) {
  int i = 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  for (i = 0; i < (int)size; i++) out[i] = (a[i] == b[i]) ? 1.0f : 0.0f;
#else
  __m256 one = _mm256_set1_ps(1.0f);
  for (; i <= (int)size - 8; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i), vb = _mm256_loadu_ps(b + i);
    _mm256_storeu_ps(out + i, _mm256_and_ps(_mm256_cmp_ps(va, vb, _CMP_EQ_OQ), one));
  }
  for (; i < (int)size; i++) out[i] = (a[i] == b[i]) ? 1.0f : 0.0f;
#endif
}

void equal_scalar_ops(float* a, float b, float* out, size_t size) {
  int i = 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  for (i = 0; i < (int)size; i++) out[i] = (a[i] == b) ? 1.0f : 0.0f;
#else
  __m256 one = _mm256_set1_ps(1.0f), vb = _mm256_set1_ps(b);
  for (; i <= (int)size - 8; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    _mm256_storeu_ps(out + i, _mm256_and_ps(_mm256_cmp_ps(va, vb, _CMP_EQ_OQ), one));
  }
  for (; i < (int)size; i++) out[i] = (a[i] == b) ? 1.0f : 0.0f;
#endif
}

void not_equal_tensor_ops(float* a, float* b, float* out, size_t size) {
  int i = 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  for (i = 0; i < (int)size; i++) out[i] = (a[i] != b[i]) ? 1.0f : 0.0f;
#else
  __m256 one = _mm256_set1_ps(1.0f);
  for (; i <= (int)size - 8; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i), vb = _mm256_loadu_ps(b + i);
    _mm256_storeu_ps(out + i, _mm256_and_ps(_mm256_cmp_ps(va, vb, _CMP_NEQ_OQ), one));
  }
  for (; i < (int)size; i++) out[i] = (a[i] != b[i]) ? 1.0f : 0.0f;
#endif
}

void not_equal_scalar_ops(float* a, float b, float* out, size_t size) {
  int i = 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  for (i = 0; i < (int)size; i++) out[i] = (a[i] != b) ? 1.0f : 0.0f;
#else
  __m256 one = _mm256_set1_ps(1.0f), vb = _mm256_set1_ps(b);
  for (; i <= (int)size - 8; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    _mm256_storeu_ps(out + i, _mm256_and_ps(_mm256_cmp_ps(va, vb, _CMP_NEQ_OQ), one));
  }
  for (; i < (int)size; i++) out[i] = (a[i] != b) ? 1.0f : 0.0f;
#endif
}

void greater_tensor_ops(float* a, float* b, float* out, size_t size) {
  int i = 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  for (i = 0; i < (int)size; i++) out[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
#else
  __m256 one = _mm256_set1_ps(1.0f);
  for (; i <= (int)size - 8; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i), vb = _mm256_loadu_ps(b + i);
    _mm256_storeu_ps(out + i, _mm256_and_ps(_mm256_cmp_ps(va, vb, _CMP_GT_OQ), one));
  }
  for (; i < (int)size; i++) out[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
#endif
}

void greater_scalar_ops(float* a, float b, float* out, size_t size) {
  int i = 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  for (i = 0; i < (int)size; i++) out[i] = (a[i] > b) ? 1.0f : 0.0f;
#else
  __m256 one = _mm256_set1_ps(1.0f), vb = _mm256_set1_ps(b);
  for (; i <= (int)size - 8; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    _mm256_storeu_ps(out + i, _mm256_and_ps(_mm256_cmp_ps(va, vb, _CMP_GT_OQ), one));
  }
  for (; i < (int)size; i++) out[i] = (a[i] > b) ? 1.0f : 0.0f;
#endif
}

void greater_equal_tensor_ops(float* a, float* b, float* out, size_t size) {
  int i = 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  for (i = 0; i < (int)size; i++) out[i] = (a[i] >= b[i]) ? 1.0f : 0.0f;
#else
  __m256 one = _mm256_set1_ps(1.0f);
  for (; i <= (int)size - 8; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i), vb = _mm256_loadu_ps(b + i);
    _mm256_storeu_ps(out + i, _mm256_and_ps(_mm256_cmp_ps(va, vb, _CMP_GE_OQ), one));
  }
  for (; i < (int)size; i++) out[i] = (a[i] >= b[i]) ? 1.0f : 0.0f;
#endif
}

void greater_equal_scalar_ops(float* a, float b, float* out, size_t size) {
  int i = 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  for (i = 0; i < (int)size; i++) out[i] = (a[i] >= b) ? 1.0f : 0.0f;
#else
  __m256 one = _mm256_set1_ps(1.0f), vb = _mm256_set1_ps(b);
  for (; i <= (int)size - 8; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    _mm256_storeu_ps(out + i, _mm256_and_ps(_mm256_cmp_ps(va, vb, _CMP_GE_OQ), one));
  }
  for (; i < (int)size; i++) out[i] = (a[i] >= b) ? 1.0f : 0.0f;
#endif
}

void smaller_tensor_ops(float* a, float* b, float* out, size_t size) {
  int i = 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  for (i = 0; i < (int)size; i++) out[i] = (a[i] < b[i]) ? 1.0f : 0.0f;
#else
  __m256 one = _mm256_set1_ps(1.0f);
  for (; i <= (int)size - 8; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i), vb = _mm256_loadu_ps(b + i);
    _mm256_storeu_ps(out + i, _mm256_and_ps(_mm256_cmp_ps(va, vb, _CMP_LT_OQ), one));
  }
  for (; i < (int)size; i++) out[i] = (a[i] < b[i]) ? 1.0f : 0.0f;
#endif
}

void smaller_scalar_ops(float* a, float b, float* out, size_t size) {
  int i = 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  for (i = 0; i < (int)size; i++) out[i] = (a[i] < b) ? 1.0f : 0.0f;
#else
  __m256 one = _mm256_set1_ps(1.0f), vb = _mm256_set1_ps(b);
  for (; i <= (int)size - 8; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    _mm256_storeu_ps(out + i, _mm256_and_ps(_mm256_cmp_ps(va, vb, _CMP_LT_OQ), one));
  }
  for (; i < (int)size; i++) out[i] = (a[i] < b) ? 1.0f : 0.0f;
#endif
}

void smaller_equal_tensor_ops(float* a, float* b, float* out, size_t size) {
  int i = 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  for (i = 0; i < (int)size; i++) out[i] = (a[i] <= b[i]) ? 1.0f : 0.0f;
#else
  __m256 one = _mm256_set1_ps(1.0f);
  for (; i <= (int)size - 8; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i), vb = _mm256_loadu_ps(b + i);
    _mm256_storeu_ps(out + i, _mm256_and_ps(_mm256_cmp_ps(va, vb, _CMP_LE_OQ), one));
  }
  for (; i < (int)size; i++) out[i] = (a[i] <= b[i]) ? 1.0f : 0.0f;
#endif
}

void smaller_equal_scalar_ops(float* a, float b, float* out, size_t size) {
  int i = 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  for (i = 0; i < (int)size; i++) out[i] = (a[i] <= b) ? 1.0f : 0.0f;
#else
  __m256 one = _mm256_set1_ps(1.0f), vb = _mm256_set1_ps(b);
  for (; i <= (int)size - 8; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    _mm256_storeu_ps(out + i, _mm256_and_ps(_mm256_cmp_ps(va, vb, _CMP_LE_OQ), one));
  }
  for (; i < (int)size; i++) out[i] = (a[i] <= b) ? 1.0f : 0.0f;
#endif
}

void transpose_1d_tensor_ops(float* a, float* out, int* shape) {
  int n = shape[0], i = 0;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  for (i = 0; i < n; i++) out[i] = a[i];
#else
  for (; i <= n - 8; i += 8) _mm256_storeu_ps(out + i, _mm256_loadu_ps(a + i));
  for (; i < n; i++) out[i] = a[i];
#endif
}

void transpose_2d_tensor_ops(float* a, float* out, int* shape) {
  int rows = shape[0], cols = shape[1];
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int idx = 0; idx < rows * cols; idx++) {
    int i = idx / cols, j = idx % cols;
    out[j * rows + i] = a[idx];
  }
}

void transpose_3d_tensor_ops(float* a, float* out, int* shape) {
  int B = shape[0], R = shape[1], C = shape[2];
  int total = B * R * C;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int idx = 0; idx < total; idx++) {
    int b = idx / (R * C), rem = idx % (R * C);
    int i = rem / C, j = rem % C;
    out[b * (C * R) + j * R + i] = a[idx];
  }
}

void transpose_ndim_tensor_ops(float* a, float* out, int* shape, int ndim) {
  size_t total_size = 1;
  for (int i = 0; i < ndim; i++) total_size *= shape[i];

  int transposed_shape[ndim];
  for (int i = 0; i < ndim; i++) transposed_shape[i] = shape[ndim - 1 - i];

#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (size_t out_idx = 0; out_idx < total_size; out_idx++) {
    size_t tmp = out_idx;
    int out_coords[ndim];
    for (int i = ndim - 1; i >= 0; i--) { out_coords[i] = tmp % transposed_shape[i]; tmp /= transposed_shape[i]; }

    size_t in_idx = 0, multiplier = 1;
    for (int i = ndim - 1; i >= 0; i--) {
      in_idx += out_coords[ndim - 1 - i] * multiplier;
      multiplier *= shape[i];
    }
    out[out_idx] = a[in_idx];
  }
}

void compute_broadcast_indices(int linear_index, int* broadcasted_shape, int max_ndim,
    int a_ndim, int b_ndim, int* a_shape, int* b_shape, int* index_a, int* index_b) {
  int strides_a[max_ndim], strides_b[max_ndim];
  int stride_a = 1, stride_b = 1;

  for (int i = max_ndim - 1; i >= 0; i--) {
    int dim_a = (i >= max_ndim - a_ndim) ? a_shape[i - (max_ndim - a_ndim)] : 1;
    int dim_b = (i >= max_ndim - b_ndim) ? b_shape[i - (max_ndim - b_ndim)] : 1;
    strides_a[i] = (dim_a == broadcasted_shape[i]) ? stride_a : 0;
    strides_b[i] = (dim_b == broadcasted_shape[i]) ? stride_b : 0;
    stride_a *= dim_a;
    stride_b *= dim_b;
  }

  *index_a = 0; *index_b = 0;
  int tmp = linear_index;
  for (int j = max_ndim - 1; j >= 0; j--) {
    int pos = tmp % broadcasted_shape[j]; tmp /= broadcasted_shape[j];
    if (strides_a[j]) *index_a += pos * strides_a[j];
    if (strides_b[j]) *index_b += pos * strides_b[j];
  }
}