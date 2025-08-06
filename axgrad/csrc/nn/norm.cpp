#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include "../cpu/ops_norm.h"
#include "norm.h"

Tensor* clip_tensor(Tensor* a, float max_val) {
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* out = (float*)malloc(a->size * sizeof(float));
  clip_tensor_ops(a_float, out, max_val, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* clamp_tensor(Tensor* a, float min_val, float max_val) {
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* out = (float*)malloc(a->size * sizeof(float));
  clamp_tensor_ops(a_float, out, min_val, max_val, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* mm_norm_tensor(Tensor* a) {
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* out = (float*)malloc(a->size * sizeof(float));
  mm_norm_tensor_ops(a_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* std_norm_tensor(Tensor* a) {
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* out = (float*)malloc(a->size * sizeof(float));
  std_norm_tensor_ops(a_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* rms_norm_tensor(Tensor* a) {
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* out = (float*)malloc(a->size * sizeof(float));
  rms_norm_tensor_ops(a_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* l1_norm_tensor(Tensor* a) {
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* out = (float*)malloc(a->size * sizeof(float));
  l1_norm_tensor_ops(a_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* l2_norm_tensor(Tensor* a) {
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* out = (float*)malloc(a->size * sizeof(float));
  l2_norm_tensor_ops(a_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* unit_norm_tensor(Tensor* a) {
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* out = (float*)malloc(a->size * sizeof(float));
  unit_norm_tensor_ops(a_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* robust_norm_tensor(Tensor* a) {
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* out = (float*)malloc(a->size * sizeof(float));
  robust_norm_tensor_ops(a_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}