/*
  - cpu.h header file for cpu.cpp & tensor.cpp
  - containing functions & ops to be preformed on cpu for Tensor
*/

#ifndef __CPU_H__
#define __CPU_H__

#include "tensor.h"

extern "C" {
  void add_tensor_cpu(Tensor* a, Tensor* b, float* out);
  void sub_tensor_cpu(Tensor* a, Tensor* b, float* out);
  void mul_tensor_cpu(Tensor* a, Tensor* b, float* out);
  void div_tensor_cpu(Tensor* a, Tensor* b, float* out);
  void add_broadcasted_tensor_cpu(Tensor* a, Tensor* b, float* out, int* broadcasted_shape, int broadcasted_size);
  void sub_broadcasted_tensor_cpu(Tensor* a, Tensor* b, float* out, int* broadcasted_shape, int broadcasted_size);
  void mul_broadcasted_tensor_cpu(Tensor* a, Tensor* b, float* out, int* broadcasted_shape, int broadcasted_size);
  void scalar_mul_tensor_cpu(Tensor* a, float b, float* out);
  void scalar_div_tensor_cpu(Tensor* a, float b, float* out);
  void tensor_div_scalar_cpu(Tensor* a, float b, float* out);
  void scalar_pow_tensor_cpu(float base, Tensor* a, float* out);
  void tensor_pow_scalar_cpu(Tensor* a, float b, float* out);
  void matmul_tensor_cpu(Tensor* a, Tensor* b, float* out);
  void broadcasted_batched_matmul_tensor_cpu(Tensor* a, Tensor* b, float* out);
  void batched_matmul_tensor_cpu(Tensor* a, Tensor* b, float* out);
  void ones_like_tensor_cpu(int size, float* out);
  void zeros_like_tensor_cpu(int size, float* out);
  void transpose_1d_tensor_cpu(Tensor* a, float* out);
  void transpose_2d_tensor_cpu(Tensor* a, float* out);
  void transpose_3d_tensor_cpu(Tensor* a, float* out);
  void reassign_tensor_cpu(Tensor* a, float* out);
  void make_contagious_tensor_cpu(Tensor* a, float* out, int* new_strides);
  void equal_tensor_cpu(Tensor* a, Tensor* b, float* out);
  void equal_broadcasted_tensor_cpu(Tensor* a, Tensor* b, float* out, int* broadcasted_shape, int broadcasted_size);

  void log_tensor_cpu(Tensor* a, float* out);
  void sum_tensor_cpu(Tensor* a, float* out, int size, int* res_shape, int axis);
  void max_tensor_cpu(Tensor* a, float* out, int size, int* res_shape, int axis);
  void min_tensor_cpu(Tensor* a, float* out, int size, int* res_shape, int axis);

  void sin_tensor_cpu(Tensor* a, float* out);
  void cos_tensor_cpu(Tensor* a, float* out);
  void sigmoid_tensor_cpu(Tensor* a, float* out);
  void tanh_tensor_cpu(Tensor* a, float* out);
  void relu_tensor_cpu(Tensor* a, float* out);
}

#endif