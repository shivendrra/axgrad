#ifndef __OPS_UNARY__H__
#define __OPS_UNARY__H__

#include <stddef.h>

extern "C" {
  void exp_tensor_ops(float* a, float* out, size_t size);
  void log_tensor_ops(float* a, float* out, size_t size);
  void abs_tensor_ops(float* a, float* out, size_t size);
  void neg_tensor_ops(float* a, float* out, size_t size);
  void sqrt_tensor_ops(float* a, float* out, size_t size);

  void sin_ops(float* a, float* out, size_t size);
  void cos_ops(float* a, float* out, size_t size);
  void tan_ops(float* a, float* out, size_t size);
  void sinh_ops(float* a, float* out, size_t size);
  void cosh_ops(float* a, float* out, size_t size);
  void tanh_ops(float* a, float* out, size_t size);
}

#endif  //!__OPS_UNARY__H__