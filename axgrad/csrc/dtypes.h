#ifndef DTYPES_H
#define DTYPES_H

#include <stdint.h>

extern "C" {
  int8_t to_int8(int value);
  int16_t to_int16(int value);
  int32_t to_int32(int value);
  int64_t to_int64(int value);
  float to_float32(float value);
  double to_float64(double value);

  void convert_dtype(const void* data, void* result, const char* dtype);
  void handle_conversion(const void* data, void* result, const char* dtype);
}

#endif