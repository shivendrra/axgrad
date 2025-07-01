#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include "cpu/ops_func.h"
#include "functional.h"

Tensor* sigmoid_tensor(Tensor* a) {
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

  sigmoid_ops(a_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }  // keep float32 or float64
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* relu_tensor(Tensor* a) {
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

  relu_ops(a_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* gelu_tensor(Tensor* a) {
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

  gelu_ops(a_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* leaky_relu_tensor(Tensor* a, float eps) {
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

  leaky_relu_ops(a_float, eps, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }  // keep float32 or float64
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* silu_tensor(Tensor* a) {
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

  silu_ops(a_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }  // keep float32 or float64
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* elu_tensor(Tensor* a, float alpha) {
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

  elu_ops(a_float, alpha, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }  // keep float32 or float64
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* swish_tensor(Tensor* a, float beta) {
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

  swish_ops(a_float, beta, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }  // keep float32 or float64
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* softplus_tensor(Tensor* a) {
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

  softplus_ops(a_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }  // keep float32 or float64
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}