#ifndef __OPS_UNARY__H__
#define __OPS_UNARY__H__

#include <stddef.h>

#define SIMD_THRESHOLD 64
#define OMP_THRESHOLD 8192
#define CACHE_CHUNK_SIZE 64

extern "C" {
  void exp_tensor_ops(float* a, float* out, size_t size);
  void log_tensor_ops(float* a, float* out, size_t size);
  void abs_tensor_ops(float* a, float* out, size_t size);
  void neg_tensor_ops(float* a, float* out, size_t size);
  void sqrt_tensor_ops(float* a, float* out, size_t size);
  void sign_tensor_ops(float* a, float* out, size_t size);

  void sin_tensor_ops(float* a, float* out, size_t size);
  void cos_tensor_ops(float* a, float* out, size_t size);
  void tan_tensor_ops(float* a, float* out, size_t size);
  void sinh_tensor_ops(float* a, float* out, size_t size);
  void cosh_tensor_ops(float* a, float* out, size_t size);
  void tanh_tensor_ops(float* a, float* out, size_t size);
}

#endif  //!__OPS_UNARY__H__