#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include "../cpu/ops_norm.h"
#include "norm.h"

Tensor* clip_tensor(Tensor* a, float max_val) {
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    exit(EXIT_FAILURE);
  }

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
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    exit(EXIT_FAILURE);
  }

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
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    exit(EXIT_FAILURE);
  }

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
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    exit(EXIT_FAILURE);
  }

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
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    exit(EXIT_FAILURE);
  }

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
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    exit(EXIT_FAILURE);
  }

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
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    exit(EXIT_FAILURE);
  }

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
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    exit(EXIT_FAILURE);
  }

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
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    exit(EXIT_FAILURE);
  }

  robust_norm_tensor_ops(a_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}