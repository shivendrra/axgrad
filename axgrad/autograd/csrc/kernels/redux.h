#ifndef __REDUX__H__
#define __REDUX__H__

#include <stddef.h>

extern "C" {
  void sum_backwards_ops(float* grad, float* out, int* original_shape, int original_ndim, size_t original_size, int axis);
  void mean_backwards_ops(float* grad, float* out, int* original_shape, int original_ndim, size_t original_size, int axis);
  void var_backwards_ops(float* a, float* grad, float* out, int* original_shape, int original_ndim, size_t original_size, int axis, int ddof);
  void std_backwards_ops(float* a, float* grad, float* out, int* original_shape, int original_ndim, size_t original_size, int axis, int ddof);
}

#endif  //!__REDUX__H__