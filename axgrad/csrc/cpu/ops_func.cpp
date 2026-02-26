#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include "ops_func.h"
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

#define C_PI 3.141592653589793f

static inline float __sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float __pdf(float x) { return (1.0f / sqrtf(2.0f * C_PI)) * expf(-0.5f * x * x); }
static inline float __erf_approx(float x) {
  float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
  int sign = x < 0 ? -1 : 1; x = fabsf(x);
  float t = 1.0f / (1.0f + p * x), y = 1.0f - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * expf(-x * x);
  return sign * y;
}
static inline float __cdf(float x) { return 0.5f * (1.0f + __erf_approx(x / sqrtf(2.0f))); }
static inline float __gelu(float x) { return x * __cdf(x); }

#if USE_AVX2
static inline void relu_chunk_avx2(float* a, float* out, size_t size) {
  const __m256 vzero = _mm256_setzero_ps();
  const size_t simd_end = size & ~7UL;
  for (size_t i = 0; i < simd_end; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    _mm256_storeu_ps(out + i, _mm256_max_ps(va, vzero));
  }
  for (size_t i = simd_end; i < size; i++) { out[i] = a[i] > 0 ? a[i] : 0; }
}

static inline void leaky_relu_chunk_avx2(float* a, float eps, float* out, size_t size) {
  const __m256 vzero = _mm256_setzero_ps(), veps = _mm256_set1_ps(eps);
  const size_t simd_end = size & ~7UL;
  for (size_t i = 0; i < simd_end; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 mask = _mm256_cmp_ps(va, vzero, _CMP_GT_OQ);
    __m256 pos = _mm256_max_ps(va, vzero), neg = _mm256_mul_ps(veps, _mm256_min_ps(va, vzero));
    _mm256_storeu_ps(out + i, _mm256_add_ps(pos, neg));
  }
  for (size_t i = simd_end; i < size; i++) { out[i] = fmaxf(0.0f, a[i]) + eps * fminf(0.0f, a[i]); }
}

static inline void relu_backwards_chunk_avx2(float* a, float* out, size_t size) {
  const __m256 vzero = _mm256_setzero_ps(), vone = _mm256_set1_ps(1.0f);
  const size_t simd_end = size & ~7UL;
  for (size_t i = 0; i < simd_end; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 mask = _mm256_cmp_ps(va, vzero, _CMP_GT_OQ);
    _mm256_storeu_ps(out + i, _mm256_and_ps(mask, vone));
  }
  for (size_t i = simd_end; i < size; i++) { out[i] = (a[i] > 0) ? 1.0f : 0.0f; }
}
#endif

void sigmoid_ops(float* a, float* out, size_t size) {
#ifdef _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = __sigmoid(a[i]); }
    return;
  }
#endif
  for (size_t i = 0; i < size; i++) { out[i] = __sigmoid(a[i]); }
}

void relu_ops(float* a, float* out, size_t size) {
#if USE_AVX2
  if (size >= SIMD_THRESHOLD) {
#ifdef _OPENMP
    if (size >= OMP_THRESHOLD) {
      #pragma omp parallel for schedule(static)
      for (size_t chunk = 0; chunk < (size + CACHE_CHUNK_SIZE - 1) / CACHE_CHUNK_SIZE; chunk++) {
        size_t start = chunk * CACHE_CHUNK_SIZE, end = start + CACHE_CHUNK_SIZE;
        if (end > size) end = size;
        relu_chunk_avx2(a + start, out + start, end - start);
      }
      return;
    }
#endif
    relu_chunk_avx2(a, out, size);
    return;
  }
#endif
#ifdef _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = a[i] > 0 ? a[i] : 0; }
    return;
  }
#endif
  for (size_t i = 0; i < size; i++) { out[i] = a[i] > 0 ? a[i] : 0; }
}

void gelu_ops(float* a, float* out, size_t size) {
#ifdef _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = __gelu(a[i]); }
    return;
  }
#endif
  for (size_t i = 0; i < size; i++) { out[i] = __gelu(a[i]); }
}

void leaky_relu_ops(float* a, float eps, float* out, size_t size) {
#if USE_AVX2
  if (size >= SIMD_THRESHOLD) {
#ifdef _OPENMP
    if (size >= OMP_THRESHOLD) {
      #pragma omp parallel for schedule(static)
      for (size_t chunk = 0; chunk < (size + CACHE_CHUNK_SIZE - 1) / CACHE_CHUNK_SIZE; chunk++) {
        size_t start = chunk * CACHE_CHUNK_SIZE, end = start + CACHE_CHUNK_SIZE;
        if (end > size) end = size;
        leaky_relu_chunk_avx2(a + start, eps, out + start, end - start);
      }
      return;
    }
#endif
    leaky_relu_chunk_avx2(a, eps, out, size);
    return;
  }
#endif
#ifdef _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = fmaxf(0.0f, a[i]) + eps * fminf(0.0f, a[i]); }
    return;
  }
#endif
  for (size_t i = 0; i < size; i++) { out[i] = fmaxf(0.0f, a[i]) + eps * fminf(0.0f, a[i]); }
}

void elu_ops(float* a, float alpha, float* out, size_t size) {
#ifdef _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = a[i] > 0 ? a[i] : alpha * (expf(a[i]) - 1.0f); }
    return;
  }
#endif
  for (size_t i = 0; i < size; i++) { out[i] = a[i] > 0 ? a[i] : alpha * (expf(a[i]) - 1.0f); }
}

void swish_ops(float* a, float beta, float* out, size_t size) {
#ifdef _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = a[i] * __sigmoid(beta * a[i]); }
    return;
  }
#endif
  for (size_t i = 0; i < size; i++) { out[i] = a[i] * __sigmoid(beta * a[i]); }
}

void silu_ops(float* a, float* out, size_t size) {
#ifdef _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = a[i] * __sigmoid(a[i]); }
    return;
  }
#endif
  for (size_t i = 0; i < size; i++) { out[i] = a[i] * __sigmoid(a[i]); }
}

void softplus_ops(float* a, float* out, size_t size) {
#ifdef _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = logf(1.0f + expf(a[i])); }
    return;
  }
#endif
  for (size_t i = 0; i < size; i++) { out[i] = logf(1.0f + expf(a[i])); }
}

void sin_backwards_ops(float* a, float* out, size_t size) {
#ifdef _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = cosf(a[i]); }
    return;
  }
#endif
  for (size_t i = 0; i < size; i++) { out[i] = cosf(a[i]); }
}

void cos_backwards_ops(float* a, float* out, size_t size) {
#ifdef _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = -sinf(a[i]); }
    return;
  }
#endif
  for (size_t i = 0; i < size; i++) { out[i] = -sinf(a[i]); }
}

void sinh_backwards_ops(float* a, float* out, size_t size) {
#ifdef _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = coshf(a[i]); }
    return;
  }
#endif
  for (size_t i = 0; i < size; i++) { out[i] = coshf(a[i]); }
}

void cosh_backwards_ops(float* a, float* out, size_t size) {
#ifdef _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = sinhf(a[i]); }
    return;
  }
#endif
  for (size_t i = 0; i < size; i++) { out[i] = sinhf(a[i]); }
}

void tanh_backwards_ops(float* a, float* out, size_t size) {
#ifdef _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = 1.0f - a[i] * a[i]; }
    return;
  }
#endif
  for (size_t i = 0; i < size; i++) { out[i] = 1.0f - a[i] * a[i]; }
}

void sigmoid_backwards_ops(float* a, float* out, size_t size) {
#ifdef _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = a[i] * (1.0f - a[i]); }
    return;
  }
#endif
  for (size_t i = 0; i < size; i++) { out[i] = a[i] * (1.0f - a[i]); }
}

void relu_backwards_ops(float* a, float* out, size_t size) {
#if USE_AVX2
  if (size >= SIMD_THRESHOLD) {
#ifdef _OPENMP
    if (size >= OMP_THRESHOLD) {
      #pragma omp parallel for schedule(static)
      for (size_t chunk = 0; chunk < (size + CACHE_CHUNK_SIZE - 1) / CACHE_CHUNK_SIZE; chunk++) {
        size_t start = chunk * CACHE_CHUNK_SIZE, end = start + CACHE_CHUNK_SIZE;
        if (end > size) end = size;
        relu_backwards_chunk_avx2(a + start, out + start, end - start);
      }
      return;
    }
#endif
    relu_backwards_chunk_avx2(a, out, size);
    return;
  }
#endif
#ifdef _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = (a[i] > 0) ? 1.0f : 0.0f; }
    return;
  }
#endif
  for (size_t i = 0; i < size; i++) { out[i] = (a[i] > 0) ? 1.0f : 0.0f; }
}

void leaky_relu_backwards_ops(float* a, float eps, float* out, size_t size) {
#ifdef _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = (a[i] > 0) ? 1.0f : eps; }
    return;
  }
#endif
  for (size_t i = 0; i < size; i++) { out[i] = (a[i] > 0) ? 1.0f : eps; }
}

void elu_backwards_ops(float* a, float alpha, float* out, size_t size) {
#ifdef _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = (a[i] > 0) ? 1.0f : alpha * expf(a[i]); }
    return;
  }
#endif
  for (size_t i = 0; i < size; i++) { out[i] = (a[i] > 0) ? 1.0f : alpha * expf(a[i]); }
}

void gelu_backwards_ops(float* a, float* out, size_t size) {
#ifdef _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = __cdf(a[i]) + a[i] * __pdf(a[i]); }
    return;
  }
#endif
  for (size_t i = 0; i < size; i++) { out[i] = __cdf(a[i]) + a[i] * __pdf(a[i]); }
}

void softplus_backwards_ops(float* a, float* out, size_t size) {
#ifdef _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) { out[i] = 1.0f / (1.0f + expf(-a[i])); }
    return;
  }
#endif
  for (size_t i = 0; i < size; i++) { out[i] = 1.0f / (1.0f + expf(-a[i])); }
}

void swish_backwards_ops(float* a, float beta, float* out, size_t size) {
#ifdef _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) {
      float x = a[i], sb = __sigmoid(beta * x);
      out[i] = sb + beta * x * sb * (1.0f - sb);
    }
    return;
  }
#endif
  for (size_t i = 0; i < size; i++) {
    float x = a[i], sb = __sigmoid(beta * x);
    out[i] = sb + beta * x * sb * (1.0f - sb);
  }
}

void silu_backwards_ops(float* a, float* out, size_t size) {
#ifdef _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) {
      float s = __sigmoid(a[i]);
      out[i] = s * (1.0f + a[i] * (1.0f - s));
    }
    return;
  }
#endif
  for (size_t i = 0; i < size; i++) {
    float s = __sigmoid(a[i]);
    out[i] = s * (1.0f + a[i] * (1.0f - s));
  }
}

void tan_backwards_ops(float* a, float* out, size_t size) {
#ifdef _OPENMP
  if (size >= OMP_THRESHOLD) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) {
      float t = tanf(a[i]);
      out[i] = 1.0f + t * t;
    }
    return;
  }
#endif
  for (size_t i = 0; i < size; i++) {
    float t = tanf(a[i]);
    out[i] = 1.0f + t * t;
  }
}