#ifndef __OPS_FUNC__H__
#define __OPS_FUNC__H__

#include <stddef.h>

#define SIMD_THRESHOLD 64
#define OMP_THRESHOLD 8192
#define CACHE_CHUNK_SIZE 64

extern "C" {
  void sigmoid_ops(float* a, float* out, size_t size);
  void relu_ops(float* a, float* out, size_t size);
  void gelu_ops(float* a, float* out, size_t size);
  void leaky_relu_ops(float* a, float eps, float* out, size_t size);
  void silu_ops(float* a, float* out, size_t size);
  void elu_ops(float* a, float alpha, float* out, size_t size);
  void swish_ops(float* a, float beta, float* out, size_t size);
  void softplus_ops(float* a, float* out, size_t size);

  void sin_backwards_ops(float* a, float* out, size_t size);
  void cos_backwards_ops(float* a, float* out, size_t size);
  void tan_backwards_ops(float* a, float* out, size_t size);
  void sinh_backwards_ops(float* a, float* out, size_t size);
  void cosh_backwards_ops(float* a, float* out, size_t size);
  void tanh_backwards_ops(float* a, float* out, size_t size);
  void sigmoid_backwards_ops(float* a, float* out, size_t size);
  void relu_backwards_ops(float* a, float* out, size_t size);
  void leaky_relu_backwards_ops(float* a, float eps, float* out, size_t size);
  void elu_backwards_ops(float* a, float alpha, float* out, size_t size);
  void silu_backwards_ops(float* a, float* out, size_t size);
  void softplus_backwards_ops(float* a, float* out, size_t size);
  void swish_backwards_ops(float* a, float beta, float* out, size_t size);
  void gelu_backwards_ops(float* a, float* out, size_t size);
}

#endif