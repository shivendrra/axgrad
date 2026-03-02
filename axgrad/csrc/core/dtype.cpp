#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <immintrin.h>
#include "dtype.h"

#pragma GCC target("avx2,fma")
#pragma GCC optimize("O3")

// dtype metadata

size_t get_dtype_size(dtype_t dtype) {
  switch (dtype) {
    case DTYPE_FLOAT32: return sizeof(float);
    case DTYPE_FLOAT64: return sizeof(double);
    case DTYPE_INT8:    return sizeof(int8_t);
    case DTYPE_INT16:   return sizeof(int16_t);
    case DTYPE_INT32:   return sizeof(int32_t);
    case DTYPE_INT64:   return sizeof(int64_t);
    case DTYPE_UINT8:   return sizeof(uint8_t);
    case DTYPE_UINT16:  return sizeof(uint16_t);
    case DTYPE_UINT32:  return sizeof(uint32_t);
    case DTYPE_UINT64:  return sizeof(uint64_t);
    case DTYPE_BOOL:    return sizeof(uint8_t);
    default:            return 0;
  }
}

const char* get_dtype_name(dtype_t dtype) {
  switch (dtype) {
    case DTYPE_FLOAT32: return "float32";
    case DTYPE_FLOAT64: return "float64";
    case DTYPE_INT8:    return "int8";
    case DTYPE_INT16:   return "int16";
    case DTYPE_INT32:   return "int32";
    case DTYPE_INT64:   return "int64";
    case DTYPE_UINT8:   return "uint8";
    case DTYPE_UINT16:  return "uint16";
    case DTYPE_UINT32:  return "uint32";
    case DTYPE_UINT64:  return "uint64";
    case DTYPE_BOOL:    return "bool";
    default:            return "unknown";
  }
}

// scalar element access

float dtype_to_float32(void* data, dtype_t dtype, size_t index) {
  switch (dtype) {
    case DTYPE_FLOAT32: return ((float*)   data)[index];
    case DTYPE_FLOAT64: return (float)((double*) data)[index];
    case DTYPE_INT8:    return (float)((int8_t*)  data)[index];
    case DTYPE_INT16:   return (float)((int16_t*) data)[index];
    case DTYPE_INT32:   return (float)((int32_t*) data)[index];
    case DTYPE_INT64:   return (float)((int64_t*) data)[index];
    case DTYPE_UINT8:   return (float)((uint8_t*)  data)[index];
    case DTYPE_UINT16:  return (float)((uint16_t*) data)[index];
    case DTYPE_UINT32:  return (float)((uint32_t*) data)[index];
    case DTYPE_UINT64:  return (float)((uint64_t*) data)[index];
    case DTYPE_BOOL:    return (float)((uint8_t*)  data)[index];
    default:            return 0.0f;
  }
}

int64_t clamp_to_int_range(double value, dtype_t dtype) {
  switch (dtype) {
    case DTYPE_INT8:
      return (value > INT8_MAX) ? INT8_MAX : (value < INT8_MIN) ? INT8_MIN : (int64_t)round(value);
    case DTYPE_INT16:
      return (value > INT16_MAX) ? INT16_MAX : (value < INT16_MIN) ? INT16_MIN : (int64_t)round(value);
    case DTYPE_INT32:
      return (value > INT32_MAX) ? INT32_MAX : (value < INT32_MIN) ? INT32_MIN : (int64_t)round(value);
    case DTYPE_INT64:
      return (value > (double)INT64_MAX) ? INT64_MAX : (value < (double)INT64_MIN) ? INT64_MIN : (int64_t)round(value);
    default:
      return (int64_t)round(value);
  }
}

uint64_t clamp_to_uint_range(double value, dtype_t dtype) {
  if (value < 0) value = 0;
  switch (dtype) {
    case DTYPE_UINT8:   return (value > UINT8_MAX)  ? UINT8_MAX  : (uint64_t)round(value);
    case DTYPE_UINT16:  return (value > UINT16_MAX) ? UINT16_MAX : (uint64_t)round(value);
    case DTYPE_UINT32:  return (value > UINT32_MAX) ? UINT32_MAX : (uint64_t)round(value);
    case DTYPE_UINT64:  return (value > (double)UINT64_MAX) ? UINT64_MAX : (uint64_t)round(value);
    case DTYPE_BOOL:    return (value != 0.0) ? 1 : 0;
    default:            return (uint64_t)round(value);
  }
}

void float32_to_dtype(float value, void* data, dtype_t dtype, size_t index) {
  switch (dtype) {
    case DTYPE_FLOAT32: ((float*)   data)[index] = value; break;
    case DTYPE_FLOAT64: ((double*)  data)[index] = (double)value; break;
    case DTYPE_INT8:    ((int8_t*)  data)[index] = (int8_t) clamp_to_int_range(value, dtype); break;
    case DTYPE_INT16:   ((int16_t*) data)[index] = (int16_t)clamp_to_int_range(value, dtype); break;
    case DTYPE_INT32:   ((int32_t*) data)[index] = (int32_t)clamp_to_int_range(value, dtype); break;
    case DTYPE_INT64:   ((int64_t*) data)[index] =          clamp_to_int_range(value, dtype); break;
    case DTYPE_UINT8:   ((uint8_t*) data)[index] = (uint8_t) clamp_to_uint_range(value, dtype); break;
    case DTYPE_UINT16:  ((uint16_t*)data)[index] = (uint16_t)clamp_to_uint_range(value, dtype); break;
    case DTYPE_UINT32:  ((uint32_t*)data)[index] = (uint32_t)clamp_to_uint_range(value, dtype); break;
    case DTYPE_UINT64:  ((uint64_t*)data)[index] =           clamp_to_uint_range(value, dtype); break;
    case DTYPE_BOOL:    ((uint8_t*) data)[index] = (uint8_t) clamp_to_uint_range(value, dtype); break;
  }
}

// AVX2 bulk conversion fast-paths
// Each specialization converts 8 elements per cycle where possible.

static void conv_f32_to_f32(const float* src, float* dst, size_t n) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(dst + i, _mm256_loadu_ps(src + i));
  for (; i < n; i++) dst[i] = src[i];
}

static void conv_f64_to_f32(const double* src, float* dst, size_t n) {
  size_t i = 0;
  // _mm256_cvtpd_ps converts 4 doubles → 4 floats
  for (; i + 4 <= n; i += 4) {
    __m256d vd = _mm256_loadu_pd(src + i);
    __m128  vf = _mm256_cvtpd_ps(vd);
    _mm_storeu_ps(dst + i, vf);
  }
  for (; i < n; i++) dst[i] = (float)src[i];
}

static void conv_i32_to_f32(const int32_t* src, float* dst, size_t n) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(dst + i, _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)(src + i))));
  for (; i < n; i++) dst[i] = (float)src[i];
}

static void conv_u8_to_f32(const uint8_t* src, float* dst, size_t n) {
  size_t i = 0;
  // _mm256_cvtepu8_epi32 zero-extends 8 uint8 → 8 int32, then cvtepi32_ps
  for (; i + 8 <= n; i += 8) {
    __m128i v8  = _mm_loadl_epi64((__m128i*)(src + i));
    __m256i v32 = _mm256_cvtepu8_epi32(v8);
    _mm256_storeu_ps(dst + i, _mm256_cvtepi32_ps(v32));
  }
  for (; i < n; i++) dst[i] = (float)src[i];
}

static void conv_i8_to_f32(const int8_t* src, float* dst, size_t n) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m128i v8  = _mm_loadl_epi64((__m128i*)(src + i));
    __m256i v32 = _mm256_cvtepi8_epi32(v8);
    _mm256_storeu_ps(dst + i, _mm256_cvtepi32_ps(v32));
  }
  for (; i < n; i++) dst[i] = (float)src[i];
}

static void conv_i16_to_f32(const int16_t* src, float* dst, size_t n) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m128i v16 = _mm_loadu_si128((__m128i*)(src + i));
    __m256i v32 = _mm256_cvtepi16_epi32(v16);
    _mm256_storeu_ps(dst + i, _mm256_cvtepi32_ps(v32));
  }
  for (; i < n; i++) dst[i] = (float)src[i];
}

static void conv_u16_to_f32(const uint16_t* src, float* dst, size_t n) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m128i v16 = _mm_loadu_si128((__m128i*)(src + i));
    __m256i v32 = _mm256_cvtepu16_epi32(v16);
    _mm256_storeu_ps(dst + i, _mm256_cvtepi32_ps(v32));
  }
  for (; i < n; i++) dst[i] = (float)src[i];
}

// convert_to_float32
// Dispatches to AVX2 fast-path where available, falls back to scalar loop.

float* convert_to_float32(void* data, dtype_t dtype, size_t size) {
  float* out = (float*)malloc(size * sizeof(float));
  if (!out) { fprintf(stderr, "Memory allocation failed\n"); return NULL; }

  switch (dtype) {
    case DTYPE_FLOAT32: conv_f32_to_f32((float*)    data, out, size); return out;
    case DTYPE_FLOAT64: conv_f64_to_f32((double*)   data, out, size); return out;
    case DTYPE_INT8:    conv_i8_to_f32 ((int8_t*)   data, out, size); return out;
    case DTYPE_INT16:   conv_i16_to_f32((int16_t*)  data, out, size); return out;
    case DTYPE_INT32:   conv_i32_to_f32((int32_t*)  data, out, size); return out;
    case DTYPE_UINT8:   conv_u8_to_f32 ((uint8_t*)  data, out, size); return out;
    case DTYPE_UINT16:  conv_u16_to_f32((uint16_t*) data, out, size); return out;
    default: break;
  }

  // Scalar fallback for int64, uint32, uint64, bool
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < size; i++) out[i] = dtype_to_float32(data, dtype, i);
  return out;
}

// convert_from_float32
// AVX2 fast-path for f32→f32 and f32→f64; scalar+OpenMP for the rest.

void convert_from_float32(float* src, void* dst, dtype_t dtype, size_t size) {
  if (dtype == DTYPE_FLOAT32) {
    conv_f32_to_f32(src, (float*)dst, size);
    return;
  }
  if (dtype == DTYPE_FLOAT64) {
    double* d = (double*)dst;
    size_t i = 0;
    // _mm256_cvtps_pd converts 4 floats → 4 doubles
    for (; i + 4 <= size; i += 4) {
      __m128  vf = _mm_loadu_ps(src + i);
      __m256d vd = _mm256_cvtps_pd(vf);
      _mm256_storeu_pd(d + i, vd);
    }
    for (; i < size; i++) d[i] = (double)src[i];
    return;
  }
  // General path — parallelise over elements
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < size; i++) float32_to_dtype(src[i], dst, dtype, i);
}

// copy_with_dtype_conversion

void copy_with_dtype_conversion(void* src, dtype_t src_dtype, void* dst, dtype_t dst_dtype, size_t size) {
  if (src_dtype == DTYPE_FLOAT32) {
    convert_from_float32((float*)src, dst, dst_dtype, size);
    return;
  }
  if (dst_dtype == DTYPE_FLOAT32) {
    float* tmp = convert_to_float32(src, src_dtype, size);
    if (tmp) { memcpy(dst, tmp, size * sizeof(float)); free(tmp); }
    return;
  }
  // General two-step conversion via float32 temp buffer
  float* tmp = convert_to_float32(src, src_dtype, size);
  if (tmp) { convert_from_float32(tmp, dst, dst_dtype, size); free(tmp); }
}

// allocate / cast

void* allocate_dtype_tensor(dtype_t dtype, size_t size) {
  void* data = malloc(size * get_dtype_size(dtype));
  if (!data) fprintf(stderr, "Memory allocation failed for dtype tensor\n");
  return data;
}

void* cast_tensor_dtype(void* data, dtype_t src_dtype, dtype_t dst_dtype, size_t size) {
  if (src_dtype == dst_dtype) {
    size_t bytes = size * get_dtype_size(src_dtype);
    void* out = malloc(bytes);
    if (!out) { fprintf(stderr, "Memory allocation failed\n"); return NULL; }
    memcpy(out, data, bytes);
    return out;
  }
  void* out = allocate_dtype_tensor(dst_dtype, size);
  if (!out) return NULL;
  copy_with_dtype_conversion(data, src_dtype, out, dst_dtype, size);
  return out;
}

// dtype predicates

int is_integer_dtype(dtype_t dtype) {
  switch (dtype) {
    case DTYPE_INT8: case DTYPE_INT16: case DTYPE_INT32: case DTYPE_INT64:
    case DTYPE_UINT8: case DTYPE_UINT16: case DTYPE_UINT32: case DTYPE_UINT64:
    case DTYPE_BOOL: return 1;
    default: return 0;
  }
}

int is_float_dtype(dtype_t dtype) {
  return (dtype == DTYPE_FLOAT32 || dtype == DTYPE_FLOAT64);
}

int is_unsigned_dtype(dtype_t dtype) {
  switch (dtype) {
    case DTYPE_UINT8: case DTYPE_UINT16: case DTYPE_UINT32: case DTYPE_UINT64:
    case DTYPE_BOOL: return 1;
    default: return 0;
  }
}

int is_signed_dtype(dtype_t dtype) {
  switch (dtype) {
    case DTYPE_INT8: case DTYPE_INT16: case DTYPE_INT32: case DTYPE_INT64:
    case DTYPE_FLOAT32: case DTYPE_FLOAT64: return 1;
    default: return 0;
  }
}

int get_dtype_priority(dtype_t dtype) {
  switch (dtype) {
    case DTYPE_BOOL:    return 1;
    case DTYPE_UINT8:   return 2;
    case DTYPE_INT8:    return 3;
    case DTYPE_UINT16:  return 4;
    case DTYPE_INT16:   return 5;
    case DTYPE_UINT32:  return 6;
    case DTYPE_INT32:   return 7;
    case DTYPE_UINT64:  return 8;
    case DTYPE_INT64:   return 9;
    case DTYPE_FLOAT32: return 10;
    case DTYPE_FLOAT64: return 11;
    default:            return 0;
  }
}

dtype_t promote_dtypes(dtype_t dtype1, dtype_t dtype2) {
  if (dtype1 == dtype2) return dtype1;
  if (is_float_dtype(dtype1) && is_integer_dtype(dtype2)) return dtype1;
  if (is_float_dtype(dtype2) && is_integer_dtype(dtype1)) return dtype2;
  if (is_float_dtype(dtype1) && is_float_dtype(dtype2))
    return (get_dtype_size(dtype1) >= get_dtype_size(dtype2)) ? dtype1 : dtype2;
  if (is_integer_dtype(dtype1) && is_integer_dtype(dtype2)) {
    if (is_signed_dtype(dtype1) != is_signed_dtype(dtype2)) {
      dtype_t u = is_unsigned_dtype(dtype1) ? dtype1 : dtype2;
      dtype_t s = is_signed_dtype(dtype1)   ? dtype1 : dtype2;
      if (get_dtype_size(u) >= get_dtype_size(s)) return u;
      size_t us = get_dtype_size(u);
      if (us <= 1) return DTYPE_INT16;
      if (us <= 2) return DTYPE_INT32;
      if (us <= 4) return DTYPE_INT64;
      return DTYPE_FLOAT64;
    }
    return (get_dtype_size(dtype1) >= get_dtype_size(dtype2)) ? dtype1 : dtype2;
  }
  return (get_dtype_priority(dtype1) >= get_dtype_priority(dtype2)) ? dtype1 : dtype2;
}