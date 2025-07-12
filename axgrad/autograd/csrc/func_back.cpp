#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include "func_back.h"
#include "../../csrc/cpu/ops_func.h"

Tensor* sin_backwards(Tensor* a) {
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

  sin_backwards_ops(a_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }  // keep float32 or float64
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* cos_backwards(Tensor* a) {
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

  cos_backwards_ops(a_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }  // keep float32 or float64
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* tan_backwards(Tensor* a) {
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

  tan_backwards_ops(a_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }  // keep float32 or float64
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* sinh_backwards(Tensor* a) {
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

  sinh_backwards_ops(a_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }  // keep float32 or float64
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* cosh_backwards(Tensor* a) {
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

  cosh_backwards_ops(a_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }  // keep float32 or float64
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* tanh_backwards(Tensor* out) {
  if (out == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* out_float = convert_to_float32(out->data, out->dtype, out->size);
  if (out_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }
  float* out_new = (float*)malloc(out->size * sizeof(float));
  if (out_new == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(out_float);
    exit(EXIT_FAILURE);
  }

  tanh_backwards_ops(out_float, out_new, out->size);
  dtype_t result_dtype;
  if (is_integer_dtype(out->dtype) || out->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = out->dtype; }  // keep float32 or float64
  Tensor* result = create_tensor(out_new, out->ndim, out->shape, out->size, result_dtype);
  free(out_float);
  free(out);
  return result;
}

Tensor* sigmoid_backwards(Tensor* out) {
  if (out == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* out_float = convert_to_float32(out->data, out->dtype, out->size);
  if (out_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }
  float* out_new = (float*)malloc(out->size * sizeof(float));
  if (out_new == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(out_float);
    exit(EXIT_FAILURE);
  }

  sigmoid_backwards_ops(out_float, out_new, out->size);
  dtype_t result_dtype;
  if (is_integer_dtype(out->dtype) || out->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = out->dtype; }  // keep float32 or float64
  Tensor* result = create_tensor(out_new, out->ndim, out->shape, out->size, result_dtype);
  free(out_float);
  free(out);
  return result;
}

Tensor* relu_backwards(Tensor* a) {
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

  relu_backwards_ops(a_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }  // keep float32 or float64
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* leaky_relu_backwards(Tensor* a, float eps) {
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

  leaky_relu_backwards_ops(a_float, eps, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }  // keep float32 or float64
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* elu_backwards(Tensor* a, float alpha) {
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

  elu_backwards_ops(a_float, alpha, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }  // keep float32 or float64
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* gelu_backwards(Tensor* a) {
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

  gelu_backwards_ops(a_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }  // keep float32 or float64
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* swish_backwards(Tensor* a, float beta) {
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

  swish_backwards_ops(a_float, beta, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }  // keep float32 or float64
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* silu_backwards(Tensor* a) {
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

  silu_backwards_ops(a_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }  // keep float32 or float64
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* softplus_backwards(Tensor* a) {
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

  softplus_backwards_ops(a_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; }
  else { result_dtype = a->dtype; }  // keep float32 or float64
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}