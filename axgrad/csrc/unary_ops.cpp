#include <stdio.h>
#include <stdlib.h>
#include "cpu/ops_unary.h"
#include "unary_ops.h"

Tensor* sin_tensor(Tensor* a) {
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

  sin_ops(a_float, out, a->size);
  // For trigonometric functions, always return float type
  // If input is integer, promote to float32; if already float, keep same precision
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) {
    result_dtype = DTYPE_FLOAT32;
  } else {
    result_dtype = a->dtype; // keep float32 or float64
  }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* sinh_tensor(Tensor* a) {
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

  sinh_ops(a_float, out, a->size);
  // For trigonometric functions, always return float type
  // If input is integer, promote to float32; if already float, keep same precision
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) {
    result_dtype = DTYPE_FLOAT32;
  } else {
    result_dtype = a->dtype; // keep float32 or float64
  }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* cos_tensor(Tensor* a) {
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

  cos_ops(a_float, out, a->size);
  // For trigonometric functions, always return float type
  // If input is integer, promote to float32; if already float, keep same precision
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) {
    result_dtype = DTYPE_FLOAT32;
  } else {
    result_dtype = a->dtype; // keep float32 or float64
  }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* cosh_tensor(Tensor* a) {
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

  cosh_ops(a_float, out, a->size);
  // For trigonometric functions, always return float type
  // If input is integer, promote to float32; if already float, keep same precision
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) {
    result_dtype = DTYPE_FLOAT32;
  } else {
    result_dtype = a->dtype; // keep float32 or float64
  }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* tan_tensor(Tensor* a) {
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

  tan_ops(a_float, out, a->size);
  // For trigonometric functions, always return float type
  // If input is integer, promote to float32; if already float, keep same precision
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) {
    result_dtype = DTYPE_FLOAT32;
  } else {
    result_dtype = a->dtype; // keep float32 or float64
  }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* tanh_tensor(Tensor* a) {
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

  tanh_ops(a_float, out, a->size);
  // For trigonometric functions, always return float type
  // If input is integer, promote to float32; if already float, keep same precision
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) {
    result_dtype = DTYPE_FLOAT32;
  } else {
    result_dtype = a->dtype; // keep float32 or float64
  }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* log_tensor(Tensor* a) {
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

  log_tensor_ops(a_float, out, a->size);
  // If input is integer, promote to float32; if already float, keep same precision
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) {
    result_dtype = DTYPE_FLOAT32;
  } else {
    result_dtype = a->dtype; // keep float32 or float64
  }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* exp_tensor(Tensor* a) {
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

  exp_tensor_ops(a_float, out, a->size);
  // If input is integer, promote to float32; if already float, keep same precision
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) {
    result_dtype = DTYPE_FLOAT32;
  } else {
    result_dtype = a->dtype; // keep float32 or float64
  }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* abs_tensor(Tensor* a) {
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

  abs_tensor_ops(a_float, out, a->size);
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, a->dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* neg_tensor(Tensor* a) {
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

  neg_tensor_ops(a_float, out, a->size);
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, a->dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* sqrt_tensor(Tensor* a) {
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

  sqrt_tensor_ops(a_float, out, a->size);
  // If input is integer, promote to float32; if already float, keep same precision
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) {
    result_dtype = DTYPE_FLOAT32;
  } else {
    result_dtype = a->dtype; // keep float32 or float64
  }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* sign_tensor(Tensor* a) {
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

  sign_tensor_ops(a_float, out, a->size);
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, a->dtype);
  free(a_float);
  free(out);
  return result;
}