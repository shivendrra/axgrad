#ifndef __OPS_BINARY__H__
#define __OPS_BINARY__H__

#include <stddef.h>

extern "C" {
  void add_ops(float* a, float* b, float* out, size_t size);
  void add_scalar_ops(float* a, float b, float* out, size_t size);
  void sub_ops(float* a, float* b, float* out, size_t size);
  void sub_scalar_ops(float* a, float b, float* out, size_t size);
  void mul_ops(float* a, float* b, float* out, size_t size);
  void mul_scalar_ops(float* a, float b, float* out, size_t size);
  void div_ops(float* a, float* b, float* out, size_t size);
  void div_scalar_ops(float* a, float b, float* out, size_t size);
  void pow_tensor_ops(float* a, float exp, float* out, size_t size);
  void pow_scalar_ops(float a, float* exp, float* out, size_t size);

  void add_broadcasted_tensor_ops(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape); 
  void sub_broadcasted_tensor_ops(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape); 
  void mul_broadcasted_tensor_ops(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape); 
  void div_broadcasted_tensor_ops(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape); 
}

#endif  //!__OPS_BINARY__H__