#ifndef __OPS_SHAPE__H__
#define __OPS_SHAPE__H__

#include <stdlib.h>

extern "C" {
  void reassign_tensor_ops(float* a, float* out, size_t size);
  void equal_tensor_ops(float* a, float* b, float* out, size_t size);
  void transpose_1d_tensor_ops(float* a, float* out, int* shape);
  void transpose_2d_tensor_ops(float* a, float* out, int* shape);
  void transpose_3d_tensor_ops(float* a, float* out, int* shape);
  void transpose_ndim_tensor_ops(float* a, float* out, int* shape, int ndim);
}

#endif