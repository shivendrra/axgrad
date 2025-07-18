#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include "basic_back.h"
#include "kernels/basic.h"
#include "../../csrc/cpu/ops_unary.h"
#include "../../csrc/cpu/ops_binary.h"
#include "../../csrc/cpu/ops_shape.h"

Tensor* log_backwards(Tensor* a, Tensor* grad) {
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
  log_backwards_ops(a_float, grad_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(grad->dtype) || grad->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(grad_float); free(a_float);
  free(out);
  return result;
}

Tensor* exp_backwards(Tensor* a, Tensor* grad) {
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
  exp_backwards_ops(a_float, grad_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(grad->dtype) || grad->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(grad_float); free(a_float);
  free(out);
  return result;
}

Tensor* sqrt_backwards(Tensor* a, Tensor* grad) {
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
  sqrt_backwards_ops(a_float, grad_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(grad->dtype) || grad->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(grad_float); free(a_float);
  free(out);
  return result;
}

Tensor* abs_backwards(Tensor* a, Tensor* grad) {
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
  abs_backwards_ops(a_float, grad_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(grad->dtype) || grad->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(grad_float); free(a_float);
  free(out);
  return result;
}

Tensor* neg_backwards(Tensor* a, Tensor* grad) {
  if (a == NULL || grad == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float *grad_float = convert_to_float32(grad->data, grad->dtype, grad->size);
  if (grad_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  neg_tensor_ops(grad_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(grad->dtype) || grad->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(grad_float);
  free(out);
  return result;
}

Tensor* add_backwards(Tensor* grad) {
  if (grad == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float *grad_float = convert_to_float32(grad->data, grad->dtype, grad->size);
  if (grad_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(grad->size * sizeof(float));
  add_scalar_backwards_ops(grad_float, out, grad->size);
  dtype_t result_dtype;
  if (is_integer_dtype(grad->dtype) || grad->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad->dtype; }
  Tensor* result = create_tensor(out, grad->ndim, grad->shape, grad->size, result_dtype);
  free(grad_float);
  free(out);
  return result;
}

Tensor* sub_backwards(Tensor* grad, bool is_first) {
  if (grad == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float *grad_float = convert_to_float32(grad->data, grad->dtype, grad->size);
  if (grad_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(grad->size * sizeof(float));
  if (is_first) {
    add_scalar_backwards_ops(grad_float, out, grad->size);
  } else {
    neg_tensor_ops(grad_float, out, grad->size);
  }
  dtype_t result_dtype;
  if (is_integer_dtype(grad->dtype) || grad->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad->dtype; }
  Tensor* result = create_tensor(out, grad->ndim, grad->shape, grad->size, result_dtype);
  free(grad_float);
  free(out);
  return result;
}

Tensor* mul_backwards(Tensor* other, Tensor* grad) {
  if (other == NULL || grad == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float *grad_float = convert_to_float32(grad->data, grad->dtype, grad->size);
  float *other_float = convert_to_float32(other->data, other->dtype, other->size);
  if (grad_float == NULL || other_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
    free(grad_float); free(other_float);
  }
  float* out = (float*)malloc(grad->size * sizeof(float));
  mul_ops(grad_float, other_float, out, grad->size);
  dtype_t result_dtype;
  if (is_integer_dtype(grad->dtype) || grad->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad->dtype; }
  Tensor* result = create_tensor(out, grad->ndim, grad->shape, grad->size, result_dtype);
  free(grad_float); free(other_float);
  free(out);
  return result;
}

Tensor* div_backwards(Tensor* a, Tensor* b, Tensor* grad, bool is_first) {
  if (a == NULL || b == NULL || grad == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float *grad_float = convert_to_float32(grad->data, grad->dtype, grad->size);
  float *a_float = convert_to_float32(a->data, a->dtype, a->size);
  float *b_float = convert_to_float32(b->data, b->dtype, b->size);
  if (grad_float == NULL || a_float == NULL || b_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
    free(grad_float); free(a_float); free(b_float);
  }
  float* out = (float*)malloc(grad->size * sizeof(float));
  if (is_first) {
    div_ops(grad_float, b_float, out, grad->size);
  } else {
    float *temp1 = (float*)malloc(grad->size * sizeof(float));
    float *temp2 = (float*)malloc(grad->size * sizeof(float));
    mul_ops(b_float, b_float, temp1, grad->size);
    div_ops(a_float, temp1, temp2, grad->size);
    neg_tensor_ops(temp2, temp2, grad->size);
    mul_ops(grad_float, temp2, out, grad->size);
    free(temp1); free(temp2);
  }
  dtype_t result_dtype;
  if (is_integer_dtype(grad->dtype) || grad->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad->dtype; }
  Tensor* result = create_tensor(out, grad->ndim, grad->shape, grad->size, result_dtype);
  free(grad_float); free(a_float); free(b_float);
  free(out);
  return result;
}

Tensor* pow_backwards(Tensor* a, float exp, Tensor* grad) {
  if (a == NULL || grad == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float *grad_float = convert_to_float32(grad->data, grad->dtype, grad->size);
  float *a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (grad_float == NULL || a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
    free(grad_float); free(a_float);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  pow_backwards_ops(a_float, exp, grad_float, out, a->size);
  dtype_t result_dtype;
  if (is_integer_dtype(grad->dtype) || grad->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad->dtype; }
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(grad_float); free(a_float);
  free(out);
  return result;
}

Tensor* add_scalar_backwards(Tensor* grad) {
  if (grad == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float *grad_float = convert_to_float32(grad->data, grad->dtype, grad->size);
  if (grad_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(grad->size * sizeof(float));
  add_scalar_backwards_ops(grad_float, out, grad->size);
  dtype_t result_dtype;
  if (is_integer_dtype(grad->dtype) || grad->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad->dtype; }
  Tensor* result = create_tensor(out, grad->ndim, grad->shape, grad->size, result_dtype);
  free(grad_float);
  free(out);
  return result;
}

Tensor* sub_scalar_backwards(Tensor* grad, bool is_first) {
  if (grad == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float *grad_float = convert_to_float32(grad->data, grad->dtype, grad->size);
  if (grad_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(grad->size * sizeof(float));
  if (is_first) {
    add_scalar_backwards_ops(grad_float, out, grad->size);
  } else {
    neg_tensor_ops(grad_float, out, grad->size);
  }
  dtype_t result_dtype;
  if (is_integer_dtype(grad->dtype) || grad->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad->dtype; }
  Tensor* result = create_tensor(out, grad->ndim, grad->shape, grad->size, result_dtype);
  free(grad_float);
  free(out);
  return result;
}

Tensor* mul_scalar_backwards(float scalar, Tensor* grad) {
  if (grad == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float *grad_float = convert_to_float32(grad->data, grad->dtype, grad->size);
  if (grad_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(grad->size * sizeof(float));
  mul_scalar_backwards_ops(scalar, grad_float, out, grad->size);
  dtype_t result_dtype;
  if (is_integer_dtype(grad->dtype) || grad->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad->dtype; }
  Tensor* result = create_tensor(out, grad->ndim, grad->shape, grad->size, result_dtype);
  free(grad_float);
  free(out);
  return result;
}

Tensor* div_scalar_backwards(float scalar, Tensor* grad, bool is_first) {
  if (grad == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float *grad_float = convert_to_float32(grad->data, grad->dtype, grad->size);
  if (grad_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(grad->size * sizeof(float));
  if (is_first) {
    div_scalar_backwards_ops(scalar, grad_float, out, grad->size);
  } else {
    float* temp = (float*)malloc(grad->size * sizeof(float));
    mul_scalar_ops(grad_float, -1.0f / (scalar * scalar), temp, grad->size);
    reassign_tensor_ops(temp, out, grad->size);
    free(temp);
  }
  dtype_t result_dtype;
  if (is_integer_dtype(grad->dtype) || grad->dtype == DTYPE_BOOL) { result_dtype = DTYPE_FLOAT32; } else { result_dtype = grad->dtype; }
  Tensor* result = create_tensor(out, grad->ndim, grad->shape, grad->size, result_dtype);
  free(grad_float);
  free(out);
  return result;
}