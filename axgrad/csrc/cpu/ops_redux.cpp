#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include "ops_redux.h"

// Helper: compute flat output index from coords, excluding axis

static inline int get_out_idx(int* coords, int* res_shape, int axis, int ndim) {
  int out_idx = 0, multiplier = 1;
  for (int d = ndim - 1; d >= 0; d--) {
    if (d != axis) {
      out_idx += coords[d] * multiplier;
      multiplier *= res_shape[ndim - 1 - (d < axis ? d : d - 1)];
    }
  }
  return out_idx;
}

void max_tensor_ops(float* a, float* out, size_t size, int* shape, int* strides, int* res_shape, int axis, int ndim) {
  if (axis == -1) {
    float max_val = a[0];
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(max:max_val)
#endif
    for (int i = 1; i < (int)size; i++) { if (a[i] > max_val) max_val = a[i]; }
    *out = max_val;
  } else {
    if (axis < 0 || axis >= ndim) { printf("Invalid axis\n"); return; }
    int out_size = 1;
    for (int i = 0; i < ndim; i++) { if (i != axis) out_size *= shape[i]; }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < out_size; i++) out[i] = -__FLT_MAX__;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < (int)size; i++) {
      int coords[ndim], tmp = i;
      for (int d = ndim - 1; d >= 0; d--) { coords[d] = tmp % shape[d]; tmp /= shape[d]; }
      int out_idx = 0, multiplier = 1;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != axis) { out_idx += coords[d] * multiplier; multiplier *= res_shape[d < axis ? d : d - 1]; }
      }
#ifdef _OPENMP
      #pragma omp atomic compare
      if (a[i] > out[out_idx]) out[out_idx] = a[i];
#else
      if (a[i] > out[out_idx]) out[out_idx] = a[i];
#endif
    }
  }
}

void min_tensor_ops(float* a, float* out, size_t size, int* shape, int* strides, int* res_shape, int axis, int ndim) {
  if (axis == -1) {
    float min_val = a[0];
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(min:min_val)
#endif
    for (int i = 1; i < (int)size; i++) { if (a[i] < min_val) min_val = a[i]; }
    *out = min_val;
  } else {
    if (axis < 0 || axis >= ndim) { printf("Invalid axis\n"); return; }
    int out_size = 1;
    for (int i = 0; i < ndim; i++) { if (i != axis) out_size *= shape[i]; }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < out_size; i++) out[i] = __FLT_MAX__;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < (int)size; i++) {
      int coords[ndim], tmp = i;
      for (int d = ndim - 1; d >= 0; d--) { coords[d] = tmp % shape[d]; tmp /= shape[d]; }
      int out_idx = 0, multiplier = 1;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != axis) { out_idx += coords[d] * multiplier; multiplier *= res_shape[d < axis ? d : d - 1]; }
      }
#ifdef _OPENMP
      #pragma omp atomic compare
      if (a[i] < out[out_idx]) out[out_idx] = a[i];
#else
      if (a[i] < out[out_idx]) out[out_idx] = a[i];
#endif
    }
  }
}

void sum_tensor_ops(float* a, float* out, int* shape, int* strides, int size, int* res_shape, int axis, int ndim) {
  if (axis == -1) {
    float sum = 0.0f;
    // AVX2 parallel accumulation
    int i = 0;
    __m256 vacc = _mm256_setzero_ps();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(+:sum)
    for (i = 0; i < size; i++) sum += a[i];
#else
    for (; i <= size - 8; i += 8) vacc = _mm256_add_ps(vacc, _mm256_loadu_ps(a + i));
    // horizontal sum of vacc
    __m128 lo = _mm256_castps256_ps128(vacc), hi = _mm256_extractf128_ps(vacc, 1);
    __m128 s  = _mm_add_ps(lo, hi); s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    sum = _mm_cvtss_f32(s);
    for (; i < size; i++) sum += a[i];
#endif
    *out = sum;
  } else {
    if (axis < 0 || axis >= ndim) { printf("Invalid Axis\n"); return; }
    int out_size = 1;
    for (int i = 0; i < ndim; i++) { if (i != axis) out_size *= shape[i]; }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < out_size; i++) out[i] = 0.0f;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < size; i++) {
      int coords[ndim], tmp = i;
      for (int d = ndim - 1; d >= 0; d--) { coords[d] = tmp % shape[d]; tmp /= shape[d]; }
      int out_idx = 0, multiplier = 1;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != axis) { out_idx += coords[d] * multiplier; multiplier *= res_shape[d < axis ? d : d - 1]; }
      }
#ifdef _OPENMP
      #pragma omp atomic
#endif
      out[out_idx] += a[i];
    }
  }
}

void mean_tensor_ops(float* a, float* out, int* shape, int* strides, int size, int* res_shape, int axis, int ndim) {
  if (axis == -1) {
    float sum = 0.0f;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(+:sum)
#endif
    for (int i = 0; i < size; i++) sum += a[i];
    *out = sum / size;
  } else {
    if (axis < 0 || axis >= ndim) { printf("Invalid Axis\n"); return; }
    int out_size = 1;
    for (int i = 0; i < ndim; i++) { if (i != axis) out_size *= shape[i]; }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < out_size; i++) out[i] = 0.0f;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < size; i++) {
      int coords[ndim], tmp = i;
      for (int d = ndim - 1; d >= 0; d--) { coords[d] = tmp % shape[d]; tmp /= shape[d]; }
      int out_idx = 0, multiplier = 1;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != axis) { out_idx += coords[d] * multiplier; multiplier *= res_shape[d < axis ? d : d - 1]; }
      }
#ifdef _OPENMP
      #pragma omp atomic
#endif
      out[out_idx] += a[i];
    }

    int axis_size = shape[axis];
    __m256 vd = _mm256_set1_ps(1.0f / axis_size);
    int i = 0;
    for (; i <= out_size - 8; i += 8)
      _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(out + i), vd));
    for (; i < out_size; i++) out[i] /= axis_size;
  }
}

void var_tensor_ops(float* a, float* out, size_t size, int* shape, int* strides, int* res_shape, int axis, int ndim, int ddof) {
  if (axis == -1) {
    float mean = 0.0f;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(+:mean)
#endif
    for (int i = 0; i < (int)size; i++) mean += a[i];
    mean /= size;

    float variance = 0.0f;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(+:variance)
#endif
    for (int i = 0; i < (int)size; i++) { float d = a[i] - mean; variance += d * d; }

    int denom = (int)size - ddof;
    if (denom <= 0) { printf("Warning: ddof >= sample size, setting variance to 0\n"); *out = 0.0f; }
    else *out = variance / denom;
  } else {
    if (axis < 0 || axis >= ndim) { printf("Invalid axis\n"); return; }
    int out_size = 1;
    for (int i = 0; i < ndim; i++) { if (i != axis) out_size *= shape[i]; }
    int axis_size = shape[axis];

    float* means = (float*)calloc(out_size, sizeof(float));
    if (!means) return;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < out_size; i++) out[i] = 0.0f;

    // Pass 1: accumulate sums for mean
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < (int)size; i++) {
      int coords[ndim], tmp = i;
      for (int d = ndim - 1; d >= 0; d--) { coords[d] = tmp % shape[d]; tmp /= shape[d]; }
      int out_idx = 0, multiplier = 1;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != axis) { out_idx += coords[d] * multiplier; multiplier *= res_shape[d < axis ? d : d - 1]; }
      }
#ifdef _OPENMP
      #pragma omp atomic
#endif
      means[out_idx] += a[i];
    }

    // AVX2 divide means
    __m256 vd = _mm256_set1_ps(1.0f / axis_size);
    int i = 0;
    for (; i <= out_size - 8; i += 8)
      _mm256_storeu_ps(means + i, _mm256_mul_ps(_mm256_loadu_ps(means + i), vd));
    for (; i < out_size; i++) means[i] /= axis_size;

    // Pass 2: accumulate squared diffs
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < (int)size; i++) {
      int coords[ndim], tmp = i;
      for (int d = ndim - 1; d >= 0; d--) { coords[d] = tmp % shape[d]; tmp /= shape[d]; }
      int out_idx = 0, multiplier = 1;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != axis) { out_idx += coords[d] * multiplier; multiplier *= res_shape[d < axis ? d : d - 1]; }
      }
      float diff = a[i] - means[out_idx];
#ifdef _OPENMP
      #pragma omp atomic
#endif
      out[out_idx] += diff * diff;
    }

    int denom = axis_size - ddof;
    if (denom <= 0) {
      printf("Warning: ddof >= sample size, setting variance to 0\n");
      for (int i = 0; i < out_size; i++) out[i] = 0.0f;
    } else {
      __m256 vden = _mm256_set1_ps(1.0f / denom);
      int i = 0;
      for (; i <= out_size - 8; i += 8)
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(out + i), vden));
      for (; i < out_size; i++) out[i] /= denom;
    }
    free(means);
  }
}

void std_tensor_ops(float* a, float* out, size_t size, int* shape, int* strides, int* res_shape, int axis, int ndim, int ddof) {
  if (axis == -1) {
    float mean = 0.0f;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(+:mean)
#endif
    for (int i = 0; i < (int)size; i++) mean += a[i];
    mean /= size;

    float variance = 0.0f;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(+:variance)
#endif
    for (int i = 0; i < (int)size; i++) { float d = a[i] - mean; variance += d * d; }

    int denom = (int)size - ddof;
    if (denom <= 0) { printf("Warning: ddof >= sample size, setting std to 0\n"); *out = 0.0f; }
    else *out = sqrtf(variance / denom);
  } else {
    if (axis < 0 || axis >= ndim) { printf("Invalid axis\n"); return; }
    int out_size = 1;
    for (int i = 0; i < ndim; i++) { if (i != axis) out_size *= shape[i]; }
    int axis_size = shape[axis];

    float* means = (float*)calloc(out_size, sizeof(float));
    if (!means) { printf("Memory allocation failed\n"); return; }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < out_size; i++) out[i] = 0.0f;

    // Pass 1: accumulate sums for mean
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < (int)size; i++) {
      int coords[ndim], tmp = i;
      for (int d = ndim - 1; d >= 0; d--) { coords[d] = tmp % shape[d]; tmp /= shape[d]; }
      int out_idx = 0, multiplier = 1;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != axis) { out_idx += coords[d] * multiplier; multiplier *= res_shape[d < axis ? d : d - 1]; }
      }
#ifdef _OPENMP
      #pragma omp atomic
#endif
      means[out_idx] += a[i];
    }

    // AVX2 divide means
    __m256 vd = _mm256_set1_ps(1.0f / axis_size);
    int i = 0;
    for (; i <= out_size - 8; i += 8)
      _mm256_storeu_ps(means + i, _mm256_mul_ps(_mm256_loadu_ps(means + i), vd));
    for (; i < out_size; i++) means[i] /= axis_size;

    // Pass 2: accumulate squared diffs
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < (int)size; i++) {
      int coords[ndim], tmp = i;
      for (int d = ndim - 1; d >= 0; d--) { coords[d] = tmp % shape[d]; tmp /= shape[d]; }
      int out_idx = 0, multiplier = 1;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != axis) { out_idx += coords[d] * multiplier; multiplier *= res_shape[d < axis ? d : d - 1]; }
      }
      float diff = a[i] - means[out_idx];
#ifdef _OPENMP
      #pragma omp atomic
#endif
      out[out_idx] += diff * diff;
    }

    int denom = axis_size - ddof;
    if (denom <= 0) {
      printf("Warning: ddof >= sample size, setting std to 0\n");
      for (int i = 0; i < out_size; i++) out[i] = 0.0f;
    } else {
      // AVX2 sqrt(x / denom)
      __m256 vden = _mm256_set1_ps(1.0f / denom);
      int i = 0;
      for (; i <= out_size - 8; i += 8)
        _mm256_storeu_ps(out + i, _mm256_sqrt_ps(_mm256_mul_ps(_mm256_loadu_ps(out + i), vden)));
      for (; i < out_size; i++) out[i] = sqrtf(out[i] / denom);
    }
    free(means);
  }
}