#ifndef __OPS_FUNC__H__
#define __OPS_FUNC__H__

#include <stddef.h>

extern "C" {
  void sigmoid_ops(float* a, float* out, size_t size);
  void relu_ops(float* a, float* out, size_t size);
  void gelu_ops(float* a, float* out, size_t size);
  void leaky_relu_ops(float* a, float eps, float* out, size_t size);
  void silu_ops(float* a, float* out, size_t size);
  void elu_ops(float* a, float alpha, float* out, size_t size);
  void swish_ops(float* a, float beta, float* out, size_t size);
  void softplus_ops(float* a, float* out, size_t size);
}

#endif