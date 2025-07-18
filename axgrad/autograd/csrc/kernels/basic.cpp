#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include "basic.h"
#include "../../../csrc/cpu/ops_binary.h"
#include "../../../csrc/cpu/ops_unary.h"
#include "../../../csrc/cpu/ops_shape.h"

void log_backwards_ops(float* a, float* grad, float* out, size_t size) { div_ops(grad, a, out, size); }
void add_scalar_backwards_ops(float* grad, float* grad_a, size_t size) { reassign_tensor_ops(grad, grad_a, size); }
void sub_scalar_backwards_ops(float* grad, float* grad_a, size_t size) { reassign_tensor_ops(grad, grad_a, size); }
void mul_scalar_backwards_ops(float scalar, float* grad, float* grad_a, size_t size) { mul_scalar_ops(grad, scalar, grad_a, size); }
void div_scalar_backwards_ops(float scalar, float* grad, float* grad_a, size_t size) { div_scalar_ops(grad, scalar, grad_a, size); }

void abs_backwards_ops(float* a, float* grad, float* out, size_t size) {
  float* temp = (float*)malloc(size * sizeof(float));
  sign_tensor_ops(a, temp, size);
  mul_ops(a, grad, out, size);
  free(temp);
}

void sqrt_backwards_ops(float* a, float* grad, float* out, size_t size) {
  float *temp1 = (float*)malloc(size * sizeof(float)), *temp2 = (float*)malloc(size * sizeof(float));
  sqrt_tensor_ops(a, temp1, size);
  div_scalar_ops(temp1, 0.5f, temp2, size);
  mul_ops(grad, temp2, out, size);
  free(temp1); free(temp2);
}

void exp_backwards_ops(float* a, float* grad, float* out, size_t size) {
  float* temp = (float*)malloc(size * sizeof(float));
  exp_tensor_ops(a, temp, size);
  mul_ops(a, grad, out, size);
}

void add_backwards_ops(float* grad, float* grad_a, float* grad_b, size_t size) {
  reassign_tensor_ops(grad, grad_a, size);
  reassign_tensor_ops(grad, grad_b, size);
}

// sub backward - grad_a = grad, grad_b = -grad
void sub_backwards_ops(float* grad, float* grad_a, float* grad_b, size_t size) {
  reassign_tensor_ops(grad, grad_a, size);
  neg_tensor_ops(grad, grad_b, size);
}

// mul backward - grad_a = grad * b, grad_b = grad * a
void mul_backwards_ops(float* a, float* b, float* grad, float* grad_a, float* grad_b, size_t size) {
  mul_ops(grad, b, grad_a, size);
  mul_ops(grad, a, grad_b, size);
}

// div backward - grad_a = grad / b, grad_b = grad * (-a / b^2)
void div_backwards_ops(float* a, float* b, float* grad, float* grad_a, float* grad_b, size_t size) {
  float *temp1 = (float*)malloc(size * sizeof(float)), *temp2 = (float*)malloc(size * sizeof(float));
  div_ops(grad, b, grad_a, size);
  mul_ops(b, b, temp1, size);
  div_ops(a, temp1, temp2, size);
  neg_tensor_ops(temp2, temp2, size);
  mul_ops(grad, temp2, grad_b, size);
  free(temp1); free(temp2);
}

// pow backward - grad_a = grad * exp * a^(exp-1)
void pow_backwards_ops(float* a, float exp, float* grad, float* grad_a, size_t size) {
  float *temp1 = (float*)malloc(size * sizeof(float)), *temp2 = (float*)malloc(size * sizeof(float));
  pow_tensor_ops(a, exp - 1.0f, temp1, size);
  mul_scalar_ops(temp1, exp, temp2, size);
  mul_ops(grad, temp2, grad_a, size);
  free(temp1); free(temp2);
}