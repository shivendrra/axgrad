#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include "ops_binary.h"
#include "ops_shape.h"

void add_ops(float* a, float* b, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = a[i] + b[i]; } }
void add_scalar_ops(float* a, float b, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = a[i] + b; } }
void sub_ops(float* a, float* b, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = a[i] - b[i]; } }
void sub_scalar_ops(float* a, float b, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = a[i] - b; } }
void mul_ops(float* a, float* b, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = a[i] * b[i]; } }
void mul_scalar_ops(float* a, float b, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = a[i] * b; } }
void div_ops(float* a, float* b, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    if (b[i] == 0.0f) {
      if (a[i] > 0.0f) { out[i] = INFINITY; }
      else if (a[i] < 0.0f) { out[i] = -INFINITY; }
      else { out[i] = NAN; } // 0/0 case
    } else { out[i] = a[i] / b[i]; }
  }
}

void div_scalar_ops(float* a, float b, float* out, size_t size) {
  if (b == 0.0f) {
    for (size_t i = 0; i < size; i++) {
      if (a[i] > 0.0f) { out[i] = INFINITY; }
      else if (a[i] < 0.0f) { out[i] = -INFINITY; }
      else { out[i] = NAN; } // 0/0 case
    } } else { for (size_t i = 0; i < size; i++) { out[i] = a[i] / b; }
  }
}

void pow_tensor_ops(float* a, float exp, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = powf(a[i], exp); } }
void pow_scalar_ops(float a, float* exp, float* out, size_t size) { for (size_t i = 0; i < size; i++) { out[i] = powf(a, exp[i]); } }

void add_broadcasted_tensor_ops(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape) {
  int max_ndim = a_ndim > b_ndim ? a_ndim : b_ndim;
  for (int i = 0; i < broadcasted_size; i++) {
    int index_a, index_b;
    compute_broadcast_indices(i, broadcasted_shape, max_ndim, a_ndim, b_ndim, a_shape, b_shape, &index_a, &index_b);
    out[i] = a[index_a] + b[index_b];
  }
}

void sub_broadcasted_tensor_ops(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape) {
  int max_ndim = a_ndim > b_ndim ? a_ndim : b_ndim;
  for (int i = 0; i < broadcasted_size; i++) {
    int index_a, index_b;
    compute_broadcast_indices(i, broadcasted_shape, max_ndim, a_ndim, b_ndim, a_shape, b_shape, &index_a, &index_b);
    out[i] = a[index_a] - b[index_b];
  }
}

void mul_broadcasted_tensor_ops(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape) {
  int max_ndim = a_ndim > b_ndim ? a_ndim : b_ndim;  // Fixed: was using min instead of max
  for (int i = 0; i < broadcasted_size; i++) {
    int index_a, index_b;
    compute_broadcast_indices(i, broadcasted_shape, max_ndim, a_ndim, b_ndim, a_shape, b_shape, &index_a, &index_b);
    out[i] = a[index_a] * b[index_b];
  }
}

void div_broadcasted_tensor_ops(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape) {
  int max_ndim = a_ndim > b_ndim ? a_ndim : b_ndim;
  for (int i = 0; i < broadcasted_size; i++) {
    int index_a, index_b;
    compute_broadcast_indices(i, broadcasted_shape, max_ndim, a_ndim, b_ndim, a_shape, b_shape, &index_a, &index_b);
    if (b[index_b] == 0.0f) {
      if (a[index_a] > 0.0f) { out[i] = INFINITY; }
      else if (a[index_a] < 0.0f) { out[i] = -INFINITY; }
      else { out[i] = NAN; }  // 0/0 case
    } else { out[i] = a[index_a] / b[index_b]; }
  }
}