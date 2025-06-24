/**
  @file red_ops.h
  @brief contains all the reduction operations helper functions
*/

#ifndef __OPS_REDUX__H__
#define __OPS_REDUX__H__

#include <stdlib.h>

extern "C" {
  void sum_tensor_ops(float* a, float* out, int* shape, int* strides, int size, int* res_shape, int axis, int ndim);
  void mean_tensor_ops(float* a, float* out, int* shape, int* strides, int size, int* res_shape, int axis, int ndim);
  void max_tensor_ops(float* a, float* out, size_t size, int* shape, int* strides, int* res_shape, int axis, int ndim);
  void min_tensor_ops(float* a, float* out, size_t size, int* shape, int* strides, int* res_shape, int axis, int ndim);
  void var_tensor_ops(float* a, float* out, size_t size, int* shape, int* strides, int* res_shape, int axis, int ndim, int ddof);
  void std_tensor_ops(float* a, float* out, size_t size, int* shape, int* strides, int* res_shape, int axis, int ndim, int ddof);
}

#endif  //!__RED_OPS__H__