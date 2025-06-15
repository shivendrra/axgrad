/**
  @file dtype.h
  @brief header file for dtype utils and value conversion
  * defines core data types used across the tensor system
  * supports dtype-specific memory handling, conversions, casting
  * used for building generic float/int/uint/boolean tensor functionality
*/

#ifndef __DTYPE__H__
#define __DTYPE__H__

#include <stdint.h>
#include <stddef.h>

// data type enumeration
typedef enum {
  DTYPE_FLOAT32,
  DTYPE_FLOAT64,
  DTYPE_INT8,
  DTYPE_INT16,
  DTYPE_INT32,
  DTYPE_INT64,
  DTYPE_UINT8,
  DTYPE_UINT16,
  DTYPE_UINT32,
  DTYPE_UINT64,
  DTYPE_BOOL
} dtype_t;

// union to hold different data types
typedef union {
  float f32;
  double f64;
  int8_t i8;
  int16_t i16;
  int32_t i32;
  int64_t i64;
  uint8_t u8;
  uint16_t u16;
  uint32_t u32;
  uint64_t u64;
  uint8_t boolean; // 0 or 1
} dtype_value_t;

extern "C" {
  size_t get_dtype_size(dtype_t dtype);   // Get size of dtype in bytes
  const char* get_dtype_name(dtype_t dtype);    // Get dtype name as string
  float dtype_to_float32(void* data, dtype_t dtype, size_t index);    // Convert any dtype value to float32 for computation
  void float32_to_dtype(float value, void* data, dtype_t dtype, size_t index);    // Convert float32 result back to original dtype
  float* convert_to_float32(void* data, dtype_t dtype, size_t size);     // Convert entire tensor from any dtype to float32
  void convert_from_float32(float* float_data, void* output_data, dtype_t dtype, size_t size);    // Convert float32 tensor back to original dtype
  void* allocate_dtype_tensor(dtype_t dtype, size_t size);  // Allocate memory for specific dtype
  void copy_with_dtype_conversion(void* src, dtype_t src_dtype, void* dst, dtype_t dst_dtype, size_t size); // Copy data with dtype conversion
  void* cast_tensor_dtype(void* data, dtype_t src_dtype, dtype_t dst_dtype, size_t size);    // Cast tensor data to different dtype

  // Helper functions for type checking and validation
  int is_integer_dtype(dtype_t dtype);
  int is_float_dtype(dtype_t dtype);
  int is_unsigned_dtype(dtype_t dtype);
  int is_signed_dtype(dtype_t dtype);

  // clamp values for integer types to prevent overflow
  int64_t clamp_to_int_range(double value, dtype_t dtype);
  uint64_t clamp_to_uint_range(double value, dtype_t dtype);
  int get_dtype_priority(dtype_t dtype);  // get the promotion priority of a dtype (higher = more preferred)
  dtype_t promote_dtypes(dtype_t dtype1, dtype_t dtype2); // promote two dtypes according to standard promotion rules
}

#endif  //!__DTYPE__H__