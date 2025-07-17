#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include "norm_back.h"
#include "kernels/norm.h"

Tensor* clip_backwards(Tensor* a, Tensor* grad, float max_val) {
  if (a == NULL || grad == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float *grad_float = convert_to_float32(grad->data, grad->dtype, grad->size), *a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (grad_float == NULL || a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
    free(grad_float); free(a_float);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  clip_backwards_ops(a_float, grad_float, out, max_val, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(grad->dtype) || grad->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(grad_float); free(a_float);
  free(out);
  return result;
}

Tensor* clamp_backwards(Tensor* a, Tensor* grad, float max_val, float min_val) {
  if (a == NULL || grad == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float *grad_float = convert_to_float32(grad->data, grad->dtype, grad->size), *a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (grad_float == NULL || a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
    free(grad_float); free(a_float);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  clamp_backwards_ops(a_float, grad_float, out, min_val, max_val, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(grad->dtype) || grad->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(grad_float); free(a_float);
  free(out);
  return result;
}

Tensor* mm_norm_backwards(Tensor* a, Tensor* grad) {
  if (a == NULL || grad == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float *grad_float = convert_to_float32(grad->data, grad->dtype, grad->size), *a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (grad_float == NULL || a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
    free(grad_float); free(a_float);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  mm_norm_backwards_ops(a_float, grad_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(grad->dtype) || grad->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(grad_float); free(a_float);
  free(out);
  return result;
}

Tensor* std_norm_backwards(Tensor* a, Tensor* grad) {
  if (a == NULL || grad == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float *grad_float = convert_to_float32(grad->data, grad->dtype, grad->size), *a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (grad_float == NULL || a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
    free(grad_float); free(a_float);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  std_norm_backwards_ops(a_float, grad_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(grad->dtype) || grad->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(grad_float); free(a_float);
  free(out);
  return result;
}

Tensor* rms_norm_backwards(Tensor* a, Tensor* grad) {
  if (a == NULL || grad == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float *grad_float = convert_to_float32(grad->data, grad->dtype, grad->size), *a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (grad_float == NULL || a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
    free(grad_float); free(a_float);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  rms_norm_backwards_ops(a_float, grad_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(grad->dtype) || grad->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(grad_float); free(a_float);
  free(out);
  return result;
}

Tensor* l1_norm_backwards(Tensor* a, Tensor* grad) {
  if (a == NULL || grad == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float *grad_float = convert_to_float32(grad->data, grad->dtype, grad->size), *a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (grad_float == NULL || a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
    free(grad_float); free(a_float);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  l1_norm_backwards_ops(a_float, grad_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(grad->dtype) || grad->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(grad_float); free(a_float);
  free(out);
  return result;
}

Tensor* l2_norm_backwards(Tensor* a, Tensor* grad) {
  if (a == NULL || grad == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float *grad_float = convert_to_float32(grad->data, grad->dtype, grad->size), *a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (grad_float == NULL || a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
    free(grad_float); free(a_float);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  l2_norm_backwards_ops(a_float, grad_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(grad->dtype) || grad->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(grad_float); free(a_float);
  free(out);
  return result;
}

Tensor* unit_norm_backwards(Tensor* a, Tensor* grad) {
  if (a == NULL || grad == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float *grad_float = convert_to_float32(grad->data, grad->dtype, grad->size), *a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (grad_float == NULL || a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
    free(grad_float); free(a_float);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  unit_norm_backwards_ops(a_float, grad_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(grad->dtype) || grad->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(grad_float); free(a_float);
  free(out);
  return result;
}

Tensor* robust_norm_backwards(Tensor* a, Tensor* grad) {
  if (a == NULL || grad == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float *grad_float = convert_to_float32(grad->data, grad->dtype, grad->size), *a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (grad_float == NULL || a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
    free(grad_float); free(a_float);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  robust_norm_backwards_ops(a_float, grad_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(grad->dtype) || grad->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(grad_float); free(a_float);
  free(out);
  return result;
}