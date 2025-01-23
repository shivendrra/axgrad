#include "dtype.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <cstdlib>

namespace axgrad {

size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::INT8: return sizeof(int8_t);
    case DType::INT16: return sizeof(int16_t);
    case DType::INT32: return sizeof(int32_t);
    case DType::INT64: return sizeof(int64_t);
    case DType::FLOAT32: return sizeof(float);
    case DType::FLOAT64: return sizeof(double);
    default: return 0;
  }
}

void* initialize_data(float value, DType dtype) {
  void* data = malloc(dtype_size(dtype));
  if (!data) {
    fprintf(stderr, "Memory allocation failed!\n");
    exit(EXIT_FAILURE);
  }
  set_data_from_float(data, dtype, value);
  return data;
}

void convert_data(void* data, DType from_dtype, DType to_dtype) {
  float value = get_data_as_float(data, from_dtype);
  set_data_from_float(data, to_dtype, value);
}

std::string dtype_to_string(DType dtype) {
  switch (dtype) {
    case DType::INT8: return "INT8";
    case DType::INT16: return "INT16";
    case DType::INT32: return "INT32";
    case DType::INT64: return "INT64";
    case DType::FLOAT32: return "FLOAT32";
    case DType::FLOAT64: return "FLOAT64";
    default: return "Unknown";
  }
}

float get_data_as_float(void* data, DType dtype) {
  switch (dtype) {
    case DType::INT8: return static_cast<float>(*reinterpret_cast<int8_t*>(data));
    case DType::INT16: return static_cast<float>(*reinterpret_cast<int16_t*>(data));
    case DType::INT32: return static_cast<float>(*reinterpret_cast<int32_t*>(data));
    case DType::INT64: return static_cast<float>(*reinterpret_cast<int64_t*>(data));
    case DType::FLOAT32: return *reinterpret_cast<float*>(data);
    case DType::FLOAT64: return static_cast<float>(*reinterpret_cast<double*>(data));
    default: return 0.0f;
  }
}

void set_data_from_float(void* data, DType dtype, float value) {
  switch (dtype) {
    case DType::INT8:
      *reinterpret_cast<int8_t*>(data) = static_cast<int8_t>(std::lround(value));
      break;
    case DType::INT16:
      *reinterpret_cast<int16_t*>(data) = static_cast<int16_t>(std::lround(value));
      break;
    case DType::INT32:
      *reinterpret_cast<int32_t*>(data) = static_cast<int32_t>(std::lround(value));
      break;
    case DType::INT64:
      *reinterpret_cast<int64_t*>(data) = static_cast<int64_t>(std::lround(value));
      break;
    case DType::FLOAT32:
      *reinterpret_cast<float*>(data) = static_cast<float>(value);
      break;
    case DType::FLOAT64:
      *reinterpret_cast<double*>(data) = static_cast<double>(value);
      break;
    default:
      fprintf(stderr, "Unknown dtype!\n");
      exit(EXIT_FAILURE);
  }
}

void free_data(void* data) {
  if (data) {
    free(data);
  }
}

}