#ifndef __BASIC__H__
#define __BASIC__H__

#include <stddef.h>

extern "C" {
  void log_backwards_ops(float* a, float* grad, float* out, size_t size);
  void abs_backwards_ops(float* a, float* grad, float* out, size_t size);
  void sqrt_backwards_ops(float* a, float* grad, float* out, size_t size);
  void exp_backwards_ops(float* a, float* grad, float* out, size_t size);
  void add_backwards_ops(float* grad, float* grad_a, float* grad_b, size_t size);
  void sub_backwards_ops(float* grad, float* grad_a, float* grad_b, size_t size);
  void mul_backwards_ops(float* a, float* b, float* grad, float* grad_a, float* grad_b, size_t size);
  void div_backwards_ops(float* a, float* b, float* grad, float* grad_a, float* grad_b, size_t size);
  void pow_backwards_ops(float* a, float exp, float* grad, float* grad_a, size_t size);
  void add_scalar_backwards_ops(float* grad, float* grad_a, size_t size);
  void sub_scalar_backwards_ops(float* grad, float* grad_a, size_t size);
  void mul_scalar_backwards_ops(float scalar, float* grad, float* grad_a, size_t size);
  void div_scalar_backwards_ops(float scalar, float* grad, float* grad_a, size_t size);
}

#endif  //!__BASIC__H__