#include <stdlib.h>
#include "helpers.h"

void ones_like_array_ops(float* out, size_t size) {
  for (int i = 0; i < size; i++) {
    out[i] = 1.0f;
  }
}

void zeros_like_array_ops(float* out, size_t size) {
  for (int i = 0; i < size; i++) {
    out[i] = 0.0f;
  }
}

void ones_array_ops(float* out, size_t size) {
  for (int i = 0; i < size; i++) {
    out[i] = 1.0f;
  }
}

void zeros_array_ops(float* out, size_t size) {
  for (int i = 0; i < size; i++) {
    out[i] = 0.0f;
  }
}

void fill_array_ops(float* out, float value, size_t size) {
  for (int i = 0; i < size; i++) {
    out[i] = value;
  }
}

void linspace_array_ops(float* out, float start, float step_size, size_t size) {
  for (int i = 0; i < size; i++) {
    out[i] = start + i * step_size;
  }
}