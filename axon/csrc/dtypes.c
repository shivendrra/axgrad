#include "dtypes.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

int8_t to_int8(int value) {
  return (int8_t)value;
}

int16_t to_int16(int value) {
  return (int16_t)value;
}

int32_t to_int32(int value) {
  return (int32_t)value;
}

int64_t to_int64(int value) {
  return (int64_t)value;
}

float to_float32(float value) {
  return value;
}

double to_float64(double value) {
  return value;
}

void convert_dtype(const void* data, void* result, const char* dtype) {
  if (strcmp(dtype, "int8") == 0) {
    *((int8_t*)result) = to_int8(*((int*)data));
  }
  else if (strcmp(dtype, "int16") == 0) {
    *((int16_t*)result) = to_int16(*((int*)data));
  }
  else if (strcmp(dtype, "int32") == 0) {
    *((int32_t*)result) = to_int32(*((int*)data));
  }
  else if (strcmp(dtype, "int64") == 0 || strcmp(dtype, "long") == 0) {
    *((int64_t*)result) = to_int64(*((int*)data));
  }
  else if (strcmp(dtype, "float32") == 0) {
    *((float*)result) = to_float32(*((float*)data));
  }
  else if (strcmp(dtype, "float64") == 0 || strcmp(dtype, "double") == 0) {
    *((double*)result) = to_float64(*((double*)data));
  }
  else {
    fprintf(stderr, "Unsupported dtype: %s\n", dtype);
    exit(EXIT_FAILURE);
  }
}

void handle_conversion(const void* data, void* result, const char* dtype) {
    // Assuming data is a flattened array of elements
    // This function needs to recursively convert data
    // For simplicity, we'll assume the data is a 1D array
  convert_dtype(data, result, dtype);
}