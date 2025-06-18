#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include "dtype.h"

size_t get_dtype_size(dtype_t dtype) {
  switch (dtype) {
    case DTYPE_FLOAT32: return sizeof(float);
    case DTYPE_FLOAT64: return sizeof(double);
    case DTYPE_INT8: return sizeof(int8_t);
    case DTYPE_INT16: return sizeof(int16_t);
    case DTYPE_INT32: return sizeof(int32_t);
    case DTYPE_INT64: return sizeof(int64_t);
    case DTYPE_UINT8: return sizeof(uint8_t);
    case DTYPE_UINT16: return sizeof(uint16_t);
    case DTYPE_UINT32: return sizeof(uint32_t);
    case DTYPE_UINT64: return sizeof(uint64_t);
    case DTYPE_BOOL: return sizeof(uint8_t);
    default: return 0;
  }
}

const char* get_dtype_name(dtype_t dtype) {
  switch (dtype) {
    case DTYPE_FLOAT32: return "float32";
    case DTYPE_FLOAT64: return "float64";
    case DTYPE_INT8: return "int8";
    case DTYPE_INT16: return "int16";
    case DTYPE_INT32: return "int32";
    case DTYPE_INT64: return "int64";
    case DTYPE_UINT8: return "uint8";
    case DTYPE_UINT16: return "uint16";
    case DTYPE_UINT32: return "uint32";
    case DTYPE_UINT64: return "uint64";
    case DTYPE_BOOL: return "bool";
    default: return "unknown";
  }
}

float dtype_to_float32(void* data, dtype_t dtype, size_t index) {
  switch (dtype) {
    case DTYPE_FLOAT32:
      return ((float*)data)[index];
    case DTYPE_FLOAT64:
      return (float)((double*)data)[index];
    case DTYPE_INT8:
      return (float)((int8_t*)data)[index];
    case DTYPE_INT16:
      return (float)((int16_t*)data)[index];
    case DTYPE_INT32:
      return (float)((int32_t*)data)[index];
    case DTYPE_INT64:
      return (float)((int64_t*)data)[index];
    case DTYPE_UINT8:
      return (float)((uint8_t*)data)[index];
    case DTYPE_UINT16:
      return (float)((uint16_t*)data)[index];
    case DTYPE_UINT32:
      return (float)((uint32_t*)data)[index];
    case DTYPE_UINT64:
      return (float)((uint64_t*)data)[index];
    case DTYPE_BOOL:
      return (float)((uint8_t*)data)[index];
    default:
      return 0.0f;
  }
}

int64_t clamp_to_int_range(double value, dtype_t dtype) {
  switch (dtype) {
    case DTYPE_INT8:
      if (value > INT8_MAX) return INT8_MAX;
      if (value < INT8_MIN) return INT8_MIN;
      return (int64_t)round(value);
    case DTYPE_INT16:
      if (value > INT16_MAX) return INT16_MAX;
      if (value < INT16_MIN) return INT16_MIN;
      return (int64_t)round(value);
    case DTYPE_INT32:
      if (value > INT32_MAX) return INT32_MAX;
      if (value < INT32_MIN) return INT32_MIN;
      return (int64_t)round(value);
    case DTYPE_INT64:
      if (value > (double)INT64_MAX) return INT64_MAX;
      if (value < (double)INT64_MIN) return INT64_MIN;
      return (int64_t)round(value);
    default:
      return (int64_t)round(value);
  }
}

uint64_t clamp_to_uint_range(double value, dtype_t dtype) {
  if (value < 0) value = 0; // Clamp negative values to 0
    
  switch (dtype) {
    case DTYPE_UINT8:
      if (value > UINT8_MAX) return UINT8_MAX;
      return (uint64_t)round(value);
    case DTYPE_UINT16:
      if (value > UINT16_MAX) return UINT16_MAX;
      return (uint64_t)round(value);
    case DTYPE_UINT32:
      if (value > UINT32_MAX) return UINT32_MAX;
      return (uint64_t)round(value);
    case DTYPE_UINT64:
      if (value > (double)UINT64_MAX) return UINT64_MAX;
      return (uint64_t)round(value);
    case DTYPE_BOOL:
      return (value != 0.0) ? 1 : 0;
    default:
      return (uint64_t)round(value);
  }
}

void float32_to_dtype(float value, void* data, dtype_t dtype, size_t index) {
  switch (dtype) {
    case DTYPE_FLOAT32:
      ((float*)data)[index] = value;
      break;
    case DTYPE_FLOAT64:
      ((double*)data)[index] = (double)value;
      break;
    case DTYPE_INT8:
      ((int8_t*)data)[index] = (int8_t)clamp_to_int_range(value, dtype);
      break;
    case DTYPE_INT16:
      ((int16_t*)data)[index] = (int16_t)clamp_to_int_range(value, dtype);
      break;
    case DTYPE_INT32:
      ((int32_t*)data)[index] = (int32_t)clamp_to_int_range(value, dtype);
      break;
    case DTYPE_INT64:
      ((int64_t*)data)[index] = clamp_to_int_range(value, dtype);
      break;
    case DTYPE_UINT8:
      ((uint8_t*)data)[index] = (uint8_t)clamp_to_uint_range(value, dtype);
      break;
    case DTYPE_UINT16:
      ((uint16_t*)data)[index] = (uint16_t)clamp_to_uint_range(value, dtype);
      break;
    case DTYPE_UINT32:
      ((uint32_t*)data)[index] = (uint32_t)clamp_to_uint_range(value, dtype);
      break;
    case DTYPE_UINT64:
      ((uint64_t*)data)[index] = clamp_to_uint_range(value, dtype);
      break;
    case DTYPE_BOOL:
      ((uint8_t*)data)[index] = (uint8_t)clamp_to_uint_range(value, dtype);
      break;
  }
}

float* convert_to_float32(void* data, dtype_t dtype, size_t size) {
  float* float_data = (float*)malloc(size * sizeof(float));
  if (float_data == NULL) {
    fprintf(stderr, "Memory allocation failed for float32 conversion\n");
    return NULL;
  }

  for (size_t i = 0; i < size; i++) {
    float_data[i] = dtype_to_float32(data, dtype, i);
  }

  return float_data;
}

void convert_from_float32(float* float_data, void* output_data, dtype_t dtype, size_t size) {
  for (size_t i = 0; i < size; i++) {
    float32_to_dtype(float_data[i], output_data, dtype, i);
  }
}

void* allocate_dtype_tensor(dtype_t dtype, size_t size) {
  size_t element_size = get_dtype_size(dtype);
  void* data = malloc(size * element_size);
  if (data == NULL) {
    fprintf(stderr, "Memory allocation failed for dtype tensor\n");
    return NULL;
  }
  return data;
}

void copy_with_dtype_conversion(void* src, dtype_t src_dtype, void* dst, dtype_t dst_dtype, size_t size) {
  for (size_t i = 0; i < size; i++) {
    float temp = dtype_to_float32(src, src_dtype, i);
    float32_to_dtype(temp, dst, dst_dtype, i);
  }
}

void* cast_tensor_dtype(void* data, dtype_t src_dtype, dtype_t dst_dtype, size_t size) {
  if (src_dtype == dst_dtype) {
    // Same dtype, just copy the data
    size_t src_size = size * get_dtype_size(src_dtype);
    void* new_data = malloc(src_size);
    if (new_data == NULL) {
      fprintf(stderr, "Memory allocation failed for tensor casting\n");
      return NULL;
    }
    memcpy(new_data, data, src_size);
    return new_data;
  }

  void* new_data = allocate_dtype_tensor(dst_dtype, size);
  if (new_data == NULL) {
    return NULL;
  }

  copy_with_dtype_conversion(data, src_dtype, new_data, dst_dtype, size);
  return new_data;
}

int is_integer_dtype(dtype_t dtype) {
  switch (dtype) {
    case DTYPE_INT8:
    case DTYPE_INT16:
    case DTYPE_INT32:
    case DTYPE_INT64:
    case DTYPE_UINT8:
    case DTYPE_UINT16:
    case DTYPE_UINT32:
    case DTYPE_UINT64:
    case DTYPE_BOOL:
      return 1;
    default:
      return 0;
  }
}

int is_float_dtype(dtype_t dtype) {
  switch (dtype) {
    case DTYPE_FLOAT32:
    case DTYPE_FLOAT64:
      return 1;
    default:
      return 0;
  }
}

int is_unsigned_dtype(dtype_t dtype) {
  switch (dtype) {
    case DTYPE_UINT8:
    case DTYPE_UINT16:
    case DTYPE_UINT32:
    case DTYPE_UINT64:
    case DTYPE_BOOL:
      return 1;
    default:
      return 0;
  }
}

int is_signed_dtype(dtype_t dtype) {
  switch (dtype) {
    case DTYPE_INT8:
    case DTYPE_INT16:
    case DTYPE_INT32:
    case DTYPE_INT64:
    case DTYPE_FLOAT32:
    case DTYPE_FLOAT64:
      return 1;
    default:
      return 0;
  }
}

int get_dtype_priority(dtype_t dtype) {
  // higher numbers = higher priority in promotion
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
  // ff same dtype, return it
  if (dtype1 == dtype2) {
    return dtype1;
  }

  // float types always win over integer types
  if (is_float_dtype(dtype1) && is_integer_dtype(dtype2)) {
    return dtype1;
  }
  if (is_float_dtype(dtype2) && is_integer_dtype(dtype1)) {
    return dtype2;
  }

  // if both are float types, choose the larger one
  if (is_float_dtype(dtype1) && is_float_dtype(dtype2)) {
    return (get_dtype_size(dtype1) >= get_dtype_size(dtype2)) ? dtype1 : dtype2;
  }

  // if both are integer types, use more complex promotion rules
  if (is_integer_dtype(dtype1) && is_integer_dtype(dtype2)) {
    // if one is signed and one is unsigned
    if (is_signed_dtype(dtype1) != is_signed_dtype(dtype2)) {
      // if unsigned type is larger or equal, use it
      dtype_t unsigned_type = is_unsigned_dtype(dtype1) ? dtype1 : dtype2;
      dtype_t signed_type = is_signed_dtype(dtype1) ? dtype1 : dtype2;
      if (get_dtype_size(unsigned_type) >= get_dtype_size(signed_type)) {
        return unsigned_type;
      } else {
        // promote to next larger signed type that can hold unsigned values
        size_t unsigned_size = get_dtype_size(unsigned_type);
        if (unsigned_size <= 1) return DTYPE_INT16;  // uint8 -> int16
        if (unsigned_size <= 2) return DTYPE_INT32;  // uint16 -> int32
        if (unsigned_size <= 4) return DTYPE_INT64;  // uint32 -> int64
        return DTYPE_FLOAT64; // uint64 -> float64 (can't fit in int64)
      }
    }

    // both have same signedness, choose larger size
    return (get_dtype_size(dtype1) >= get_dtype_size(dtype2)) ? dtype1 : dtype2;
  }

  // fallback: use priority system
  return (get_dtype_priority(dtype1) >= get_dtype_priority(dtype2)) ? dtype1 : dtype2;
}