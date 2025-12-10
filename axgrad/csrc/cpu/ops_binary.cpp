#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include "ops_shape.h"
#include "ops_binary.h"

void add_ops_avx2(float* a, float* b, float* out, size_t size) {
  for (size_t i = 0; i < size; i += 8) {
    if (i + 8 <= size) {
      __m256 va = _mm256_loadu_ps(a + i);
      __m256 vb = _mm256_loadu_ps(b + i);
      _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
    } else {
      for (size_t j = i; j < size; j++) out[j] = a[j] + b[j];
      break;
    }
  }
}

void sub_ops_avx2(float* a, float* b, float* out, size_t size) {
  for (size_t i = 0; i < size; i += 8) {
    if (i + 8 <= size) {
      __m256 va = _mm256_loadu_ps(a + i);
      __m256 vb = _mm256_loadu_ps(b + i);
      _mm256_storeu_ps(out + i, _mm256_sub_ps(va, vb));
    } else {
      for (size_t j = i; j < size; j++) out[j] = a[j] - b[j];
      break;
    }
  }
}

void mul_ops_avx2(float* a, float* b, float* out, size_t size) {
  for (size_t i = 0; i < size; i += 8) {
    if (i + 8 <= size) {
      __m256 va = _mm256_loadu_ps(a + i);
      __m256 vb = _mm256_loadu_ps(b + i);
      _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
    } else {
      for (size_t j = i; j < size; j++) out[j] = a[j] * b[j];
      break;
    }
  }
}

void div_ops_avx2(float* a, float* b, float* out, size_t size) {
  for (size_t i = 0; i < size; i += 8) {
    if (i + 8 <= size) {
      __m256 va = _mm256_loadu_ps(a + i);
      __m256 vb = _mm256_loadu_ps(b + i);
      _mm256_storeu_ps(out + i, _mm256_div_ps(va, vb));
    } else {
      for (size_t j = i; j < size; j++) {
        if (b[j] == 0.0f) {
          if (a[j] > 0.0f) out[j] = INFINITY;
          else if (a[j] < 0.0f) out[j] = -INFINITY;
          else out[j] = NAN;
        } else out[j] = a[j] / b[j];
      }
      break;
    }
  }
}

void add_ops_hybrid(float* a, float* b, float* out, size_t size) {
  #pragma omp parallel for schedule(static)
  for (size_t chunk = 0; chunk < (size + 63) / 64; chunk++) {
    size_t start = chunk * 64, end = (start + 64 < size) ? start + 64 : size;
    add_ops_avx2(a + start, b + start, out + start, end - start);
  }
}

void sub_ops_hybrid(float* a, float* b, float* out, size_t size) {
  #pragma omp parallel for schedule(static)
  for (size_t chunk = 0; chunk < (size + 63) / 64; chunk++) {
    size_t start = chunk * 64, end = (start + 64 < size) ? start + 64 : size;
    sub_ops_avx2(a + start, b + start, out + start, end - start);
  }
}

void mul_ops_hybrid(float* a, float* b, float* out, size_t size) {
  #pragma omp parallel for schedule(static)
  for (size_t chunk = 0; chunk < (size + 63) / 64; chunk++) {
    size_t start = chunk * 64, end = (start + 64 < size) ? start + 64 : size;
    mul_ops_avx2(a + start, b + start, out + start, end - start);
  }
}

void div_ops_hybrid(float* a, float* b, float* out, size_t size) {
  #pragma omp parallel for schedule(static)
  for (size_t chunk = 0; chunk < (size + 63) / 64; chunk++) {
    size_t start = chunk * 64, end = (start + 64 < size) ? start + 64 : size;
    div_ops_avx2(a + start, b + start, out + start, end - start);
  }
}

void add_ops(float* a, float* b, float* out, size_t size) {
  if (size >= 64) { add_ops_hybrid(a, b, out, size); return; }
  add_ops_avx2(a, b, out, size);
}

void add_scalar_ops(float* a, float b, float* out, size_t size) {
  __m256 vb = _mm256_set1_ps(b);
  #pragma omp parallel for schedule(static) if(size >= 1024)
  for (size_t i = 0; i < size; i += 8) {
    if (i + 8 <= size) {
      __m256 va = _mm256_loadu_ps(a + i);
      _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
    } else {
      for (size_t j = i; j < size; j++) out[j] = a[j] + b;
    }
  }
}

void sub_ops(float* a, float* b, float* out, size_t size) {
  if (size >= 64) { sub_ops_hybrid(a, b, out, size); return; }
  sub_ops_avx2(a, b, out, size);
}

void sub_scalar_ops(float* a, float b, float* out, size_t size) {
  __m256 vb = _mm256_set1_ps(b);
  #pragma omp parallel for schedule(static) if(size >= 1024)
  for (size_t i = 0; i < size; i += 8) {
    if (i + 8 <= size) {
      __m256 va = _mm256_loadu_ps(a + i);
      _mm256_storeu_ps(out + i, _mm256_sub_ps(va, vb));
    } else {
      for (size_t j = i; j < size; j++) out[j] = a[j] - b;
    }
  }
}

void mul_ops(float* a, float* b, float* out, size_t size) {
  if (size >= 64) { mul_ops_hybrid(a, b, out, size); return; }
  mul_ops_avx2(a, b, out, size);
}

void mul_scalar_ops(float* a, float b, float* out, size_t size) {
  __m256 vb = _mm256_set1_ps(b);
  #pragma omp parallel for schedule(static) if(size >= 1024)
  for (size_t i = 0; i < size; i += 8) {
    if (i + 8 <= size) {
      __m256 va = _mm256_loadu_ps(a + i);
      _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
    } else {
      for (size_t j = i; j < size; j++) out[j] = a[j] * b;
    }
  }
}

void div_ops(float* a, float* b, float* out, size_t size) {
  if (size >= 64) { div_ops_hybrid(a, b, out, size); return; }
  div_ops_avx2(a, b, out, size);
}

void div_scalar_ops(float* a, float b, float* out, size_t size) {
  if (b == 0.0f) {
    #pragma omp parallel for schedule(static) if(size >= 1024)
    for (size_t i = 0; i < size; i++) {
      if (a[i] > 0.0f) out[i] = INFINITY;
      else if (a[i] < 0.0f) out[i] = -INFINITY;
      else out[i] = NAN;
    }
  } else {
    __m256 vb = _mm256_set1_ps(b);
    #pragma omp parallel for schedule(static) if(size >= 1024)
    for (size_t i = 0; i < size; i += 8) {
      if (i + 8 <= size) {
        __m256 va = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(out + i, _mm256_div_ps(va, vb));
      } else {
        for (size_t j = i; j < size; j++) out[j] = a[j] / b;
      }
    }
  }
}

void pow_tensor_ops(float* a, float exp, float* out, size_t size) {
  #pragma omp parallel for schedule(static) if(size >= 1024)
  for (size_t i = 0; i < size; i++) out[i] = powf(a[i], exp);
}

void pow_scalar_ops(float a, float* exp, float* out, size_t size) {
  #pragma omp parallel for schedule(static) if(size >= 1024)
  for (size_t i = 0; i < size; i++) out[i] = powf(a, exp[i]);
}

void add_broadcasted_tensor_ops(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape) {
  int max_ndim = a_ndim > b_ndim ? a_ndim : b_ndim;
  #pragma omp parallel for schedule(static) if(broadcasted_size >= 1024)
  for (int i = 0; i < broadcasted_size; i++) {
    int index_a, index_b;
    compute_broadcast_indices(i, broadcasted_shape, max_ndim, a_ndim, b_ndim, a_shape, b_shape, &index_a, &index_b);
    out[i] = a[index_a] + b[index_b];
  }
}

void sub_broadcasted_tensor_ops(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape) {
  int max_ndim = a_ndim > b_ndim ? a_ndim : b_ndim;
  #pragma omp parallel for schedule(static) if(broadcasted_size >= 1024)
  for (int i = 0; i < broadcasted_size; i++) {
    int index_a, index_b;
    compute_broadcast_indices(i, broadcasted_shape, max_ndim, a_ndim, b_ndim, a_shape, b_shape, &index_a, &index_b);
    out[i] = a[index_a] - b[index_b];
  }
}

void mul_broadcasted_tensor_ops(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape) {
  int max_ndim = a_ndim > b_ndim ? a_ndim : b_ndim;
  #pragma omp parallel for schedule(static) if(broadcasted_size >= 1024)
  for (int i = 0; i < broadcasted_size; i++) {
    int index_a, index_b;
    compute_broadcast_indices(i, broadcasted_shape, max_ndim, a_ndim, b_ndim, a_shape, b_shape, &index_a, &index_b);
    out[i] = a[index_a] * b[index_b];
  }
}

void div_broadcasted_tensor_ops(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape) {
  int max_ndim = a_ndim > b_ndim ? a_ndim : b_ndim;
  #pragma omp parallel for schedule(static) if(broadcasted_size >= 1024)
  for (int i = 0; i < broadcasted_size; i++) {
    int index_a, index_b;
    compute_broadcast_indices(i, broadcasted_shape, max_ndim, a_ndim, b_ndim, a_shape, b_shape, &index_a, &index_b);
    if (b[index_b] == 0.0f) {
      if (a[index_a] > 0.0f) out[i] = INFINITY;
      else if (a[index_a] < 0.0f) out[i] = -INFINITY;
      else out[i] = NAN;
    } else out[i] = a[index_a] / b[index_b];
  }
}