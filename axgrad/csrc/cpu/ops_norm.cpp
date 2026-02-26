#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "ops_norm.h"
#include "ops_redux.h"
#include "ops_binary.h"
#include "ops_unary.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2 1
#else
#define USE_AVX2 0
#endif

static inline void __mm_norm(float* a, float* out, size_t size) {
  float min_val = a[0], max_val = a[0];

#ifdef _OPENMP
  #pragma omp parallel for schedule(static) reduction(min:min_val) reduction(max:max_val)
#endif
  for (size_t i = 1; i < size; i++) {
    if (a[i] < min_val) min_val = a[i];
    if (a[i] > max_val) max_val = a[i];
  }

  float range = max_val - min_val;
  if (range == 0.0f) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < size; i++) out[i] = 0.0f;
  } else {
    sub_scalar_ops(a, min_val, out, size);
    div_scalar_ops(out, range, out, size);
  }
}

static inline void __std_norm(float* a, float* out, size_t size) {
  float sum = 0.0f;

#ifdef _OPENMP
  #pragma omp parallel for schedule(static) reduction(+:sum)
#endif
  for (size_t i = 0; i < size; i++) sum += a[i];
  float mean = sum / (float)size;

  float var_sum = 0.0f;

#ifdef _OPENMP
  #pragma omp parallel for schedule(static) reduction(+:var_sum)
#endif
  for (size_t i = 0; i < size; i++) {
    float diff = a[i] - mean;
    var_sum += diff * diff;
  }

  float std_dev = sqrtf(var_sum / (float)size);

  if (std_dev == 0.0f) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < size; i++) out[i] = 0.0f;
  } else {
    sub_scalar_ops(a, mean, out, size);
    div_scalar_ops(out, std_dev, out, size);
  }
}

void clip_tensor_ops(float* a, float* out, float max_val, size_t size) {
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < size; i++)
    out[i] = (a[i] > max_val) ? max_val : ((a[i] < -max_val) ? -max_val : a[i]);
}

void clamp_tensor_ops(float* a, float* out, float min_val, float max_val, size_t size) {
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < size; i++)
    out[i] = (a[i] > max_val) ? max_val : ((a[i] < min_val) ? min_val : a[i]);
}

void mm_norm_tensor_ops(float* a, float* out, size_t size) {
  __mm_norm(a, out, size);
}

void std_norm_tensor_ops(float* a, float* out, size_t size) {
  __std_norm(a, out, size);
}

void rms_norm_tensor_ops(float* a, float* out, size_t size) {
  float sum = 0.0f;

#ifdef _OPENMP
  #pragma omp parallel for schedule(static) reduction(+:sum)
#endif
  for (size_t i = 0; i < size; i++) sum += a[i] * a[i];

  float rms = sqrtf(sum / (float)size);
  if (rms == 0.0f) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < size; i++) out[i] = 0.0f;
  } else { div_scalar_ops(a, rms, out, size); }
}

void l1_norm_tensor_ops(float* a, float* out, size_t size) {
  float sum = 0.0f;

#ifdef _OPENMP
  #pragma omp parallel for schedule(static) reduction(+:sum)
#endif
  for (size_t i = 0; i < size; i++) sum += fabsf(a[i]);

  if (sum == 0.0f) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < size; i++) out[i] = 0.0f;
  } else { div_scalar_ops(a, sum, out, size); }
}

void l2_norm_tensor_ops(float* a, float* out, size_t size) {
  float sum = 0.0f;

#ifdef _OPENMP
  #pragma omp parallel for schedule(static) reduction(+:sum)
#endif
  for (size_t i = 0; i < size; i++) sum += a[i] * a[i];

  float l2_norm = sqrtf(sum);
  if (l2_norm == 0.0f) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < size; i++) out[i] = 0.0f;
  } else { div_scalar_ops(a, l2_norm, out, size); }
}

void unit_norm_tensor_ops(float* a, float* out, size_t size) { l2_norm_tensor_ops(a, out, size); }

// Comparison function for qsort
static int _cmp_float(const void* x, const void* y) {
  float fx = *(const float*)x, fy = *(const float*)y;
  return (fx > fy) - (fx < fy);
}

void robust_norm_tensor_ops(float* a, float* out, size_t size) {
  float* temp = (float*)malloc(size * sizeof(float));
  if (!temp) return;

  // Copy & sort to find median
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < size; i++) temp[i] = a[i];
  qsort(temp, size, sizeof(float), _cmp_float);

  float median = (size % 2 == 0)
    ? (temp[size/2 - 1] + temp[size/2]) / 2.0f
    : temp[size/2];

  // Compute absolute deviations in parallel, then sort for MAD
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < size; i++) temp[i] = fabsf(a[i] - median);
  qsort(temp, size, sizeof(float), _cmp_float);

  float mad = (size % 2 == 0)
    ? (temp[size/2 - 1] + temp[size/2]) / 2.0f
    : temp[size/2];

  if (mad == 0.0f) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < size; i++) out[i] = 0.0f;
  } else {
    sub_scalar_ops(a, median, out, size);
    div_scalar_ops(out, mad, out, size);
  }
  free(temp);
}