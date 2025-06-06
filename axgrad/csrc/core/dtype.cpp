/*
  * dtype.cpp main file that contains all dtype related ops
  * changes the dtype of each scalar & tensor values, up-castes & re-castes
    values: float->dtype->float as needed.
*/

#include "dtype.h"
#include <iostream>
#include <string.h>
#include <math.h>
#include <stdint.h>

size_t dtype_size(DType dtype) {
  switch (dtype)
  {
  case DType::INT8: return sizeof(int8_t);
  case DType::INT16: return sizeof(int16_t);
  case DType::INT32: return sizeof(int32_t);
  case DType::INT64: return sizeof(int64_t);
  case DType::FLOAT32: return sizeof(float);
  case DType::FLOAT64: return sizeof(double);
  default: return 0;
  }
}

void* init_data(float* values, DType dtype, int size) {
  if (!values || size <= 0) return NULL;
  size_t type_size = dtype_size(dtype);
  if (type_size == 0) {
    fprintf(stderr, "Invalid dtype!\n");
    return NULL;
  }

  void* data = malloc(size * type_size);
  if (!data) {
    fprintf(stderr, "Memory allocation failed!\n");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < size; i++) {
    set_data_from_float((char*)data + i * type_size, dtype, values[i]);
  }
  return data;
}

const char* dtype_to_string(DType dtype) {
  switch (dtype) {
    case DType::INT8: return "INT8";
    case DType::INT16: return "INT16";
    case DType::INT32: return "INT32";
    case DType::INT64: return "INT64";
    case DType::FLOAT32: return "FLOAT32";
    case DType::FLOAT64: return "FLOAT64";
    default: return "UNKNOWN!";
  }
}

void convert_data(void* data, DType from_dtype, DType to_dtype) {
  if (!data) return;
  float temp_value = get_data_as_float(data, from_dtype);
  set_data_from_float(data, to_dtype, temp_value);
}

float get_data_as_float(void* data, DType dtype) {
  if (!data) return 0.0;
  switch (dtype) {
    case DType::INT8: return *reinterpret_cast<int8_t*>(data);
    case DType::INT16: return *reinterpret_cast<int16_t*>(data);
    case DType::INT32: return *reinterpret_cast<int32_t*>(data);
    case DType::INT64: return *reinterpret_cast<int64_t*>(data);
    case DType::FLOAT32: return *reinterpret_cast<float*>(data);
    case DType::FLOAT64: return *reinterpret_cast<double*>(data);
    default: return 0.0;
  }
}

void set_data_from_float(void* data, DType dtype, float value) {
  if (!data) return;
  switch (dtype) {
    case DType::INT8:
      *reinterpret_cast<int8_t*>(data) = static_cast<int8_t>(round(value));
      break;
    case DType::INT16:
      *reinterpret_cast<int16_t*>(data) = static_cast<int16_t>(round(value));
      break;
    case DType::INT32:
      *reinterpret_cast<int32_t*>(data) = static_cast<int32_t>(round(value));
      break;
    case DType::INT64:
      *reinterpret_cast<int64_t*>(data) = static_cast<int64_t>(round(value));
      break;
    case DType::FLOAT32:
      *reinterpret_cast<float*>(data) = static_cast<float>(value);
      break;
    case DType::FLOAT64:
      *reinterpret_cast<double*>(data) = static_cast<double>(value);
      break;
    default:
      printf("Unknown type!\n");
  }
}

void free_data(void* data) {
  if (data) {
    free(data);
  }
}