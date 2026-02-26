#include <stdio.h>
#include <stddef.h>
#include <math.h>
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

void exp_tensor_ops(float* a, float* out, size_t size) {
#ifdef  _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = expf(a[i]); }
    return;
  }
#endif  //_OPENMP
  for (size_t i = 0; i < size; i++) { out[i] = expf(a[i]); }
}

void log_tensor_ops(float* a, float* out, size_t size) {
#ifdef  _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = logf(a[i]); }
    return;
  }
#endif  //_OPENMP
  for (size_t i = 0; i < size; i++) { out[i] = logf(a[i]); }
}

void abs_tensor_ops(float* a, float* out, size_t size) {
#ifdef  _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = fabsf(a[i]); }
    return;
  }
#endif  //_OPENMP
  for (size_t i = 0; i < size; i++) { out[i] = fabsf(a[i]); }
}

void sin_tensor_ops(float* a, float* out, size_t size) {
#ifdef  _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = sinf(a[i]); }
    return;
  }
#endif  //_OPENMP
  for (size_t i = 0; i < size; i++) { out[i] = sinf(a[i]); }
}

void cos_tensor_ops(float* a, float* out, size_t size) {
#ifdef  _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = cosf(a[i]); }
    return;
  }
#endif  //_OPENMP
  for (size_t i = 0; i < size; i++) { out[i] = cosf(a[i]); }
}

void tan_tensor_ops(float* a, float* out, size_t size) {
#ifdef  _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = tanf(a[i]); }
    return;
  }
#endif  //_OPENMP
  for (size_t i = 0; i < size; i++) { out[i] = tanf(a[i]); }
}

void sinh_tensor_ops(float* a, float* out, size_t size) {
#ifdef  _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = sinhf(a[i]); }
    return;
  }
#endif  //_OPENMP
  for (size_t i = 0; i < size; i++) { out[i] = sinhf(a[i]); }
}

void cosh_tensor_ops(float* a, float* out, size_t size) {
#ifdef  _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = coshf(a[i]); }
    return;
  }
#endif  //_OPENMP
  for (size_t i = 0; i < size; i++) { out[i] = coshf(a[i]); }
}

void tanh_tensor_ops(float* a, float* out, size_t size) {
#ifdef  _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = tanhf(a[i]); }
    return;
  }
#endif  //_OPENMP
  for (size_t i = 0; i < size; i++) { out[i] = tanhf(a[i]); }
}

void sqrt_tensor_ops(float* a, float* out, size_t size) {
#ifdef  _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = sqrtf(a[i]); }
    return;
  }
#endif  //_OPENMP
  for (size_t i = 0; i < size; i++) { out[i] = sqrtf(a[i]); }
}

void neg_tensor_ops(float* a, float* out, size_t size) {
#ifdef  _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = -a[i]; }
    return;
  }
#endif  //_OPENMP
  for (size_t i = 0; i < size; i++) { out[i] = -a[i]; }
}

void sign_tensor_ops(float* a, float* out, size_t size) {
#ifdef  _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = (a[i] > 0) ? 1.0f : ((a[i] < 0) ? -1.0f : 0.0f); }
    return;
  }
#endif  //_OPENMP
  for (size_t i = 0; i < size; i++) { out[i] = (a[i] > 0) ? 1.0f : ((a[i] < 0) ? -1.0f : 0.0f); }
}