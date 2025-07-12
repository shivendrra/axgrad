#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include "redux_back.h"
#include "kernels/redux.h"
#include "../../csrc/cpu/helpers.h"

Tensor* sum_backwards(Tensor* grad_output, int* original_shape, int original_ndim, size_t original_size, int axis) {
  if (grad_output == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* grad_float = convert_to_float32(grad_output->data, grad_output->dtype, grad_output->size);
  if (grad_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(original_size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(grad_float);
    exit(EXIT_FAILURE);
  }

  sum_backwards_ops(grad_float, out, original_shape, original_ndim, original_size, axis);
  dtype_t result_dtype;
  if (is_integer_dtype(grad_output->dtype) || grad_output->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad_output->dtype; }
  Tensor* result = create_tensor(out, original_ndim, original_shape, original_size, result_dtype);
  free(grad_float);
  free(out);
  return result;
}

Tensor* mean_backwards(Tensor* grad_output, int* original_shape, int original_ndim, size_t original_size, int axis) {
  if (grad_output == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* grad_float = convert_to_float32(grad_output->data, grad_output->dtype, grad_output->size);
  if (grad_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(original_size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(grad_float);
    exit(EXIT_FAILURE);
  }

  mean_backwards_ops(grad_float, out, original_shape, original_ndim, original_size, axis);
  dtype_t result_dtype;
  if (is_integer_dtype(grad_output->dtype) || grad_output->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad_output->dtype; }
  Tensor* result = create_tensor(out, original_ndim, original_shape, original_size, result_dtype);
  free(grad_float);
  free(out);
  return result;
}

Tensor* var_backwards(Tensor* a, Tensor* grad_output, int* original_shape, int original_ndim, size_t original_size, int axis, int ddof) {
  if (a == NULL || grad_output == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }
  float* grad_float = convert_to_float32(grad_output->data, grad_output->dtype, grad_output->size);
  if (grad_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    free(a_float);
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(original_size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    free(grad_float);
    exit(EXIT_FAILURE);
  }

  var_backwards_ops(a_float, grad_float, out, original_shape, original_ndim, original_size, axis, ddof);
  dtype_t result_dtype;
  if (is_integer_dtype(grad_output->dtype) || grad_output->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad_output->dtype; }
  Tensor* result = create_tensor(out, original_ndim, original_shape, original_size, result_dtype);
  free(a_float);
  free(grad_float);
  free(out);
  return result;
}

Tensor* std_backwards(Tensor* a, Tensor* grad_output, int* original_shape, int original_ndim, size_t original_size, int axis, int ddof) {
  if (a == NULL || grad_output == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }
  float* grad_float = convert_to_float32(grad_output->data, grad_output->dtype, grad_output->size);
  if (grad_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    free(a_float);
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(original_size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    free(grad_float);
    exit(EXIT_FAILURE);
  }

  std_backwards_ops(a_float, grad_float, out, original_shape, original_ndim, original_size, axis, ddof);
  dtype_t result_dtype;
  if (is_integer_dtype(grad_output->dtype) || grad_output->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad_output->dtype; }
  Tensor* result = create_tensor(out, original_ndim, original_shape, original_size, result_dtype);
  free(a_float);
  free(grad_float);
  free(out);
  return result;
}