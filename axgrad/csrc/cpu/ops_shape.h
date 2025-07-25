#ifndef __OPS_SHAPE__H__
#define __OPS_SHAPE__H__

#include <stdlib.h>

extern "C" {
  void reassign_tensor_ops(float* a, float* out, size_t size);
  void equal_tensor_ops(float* a, float* b, float* out, size_t size);
  void equal_scalar_ops(float* a, float b, float* out, size_t size);
  void not_equal_tensor_ops(float* a, float* b, float* out, size_t size);
  void not_equal_scalar_ops(float* a, float b, float* out, size_t size);
  void greater_tensor_ops(float* a, float* b, float* out, size_t size);
  void greater_scalar_ops(float* a, float b, float* out, size_t size);
  void greater_equal_tensor_ops(float* a, float* b, float* out, size_t size);
  void greater_equal_scalar_ops(float* a, float b, float* out, size_t size);
  void smaller_tensor_ops(float* a, float* b, float* out, size_t size);
  void smaller_scalar_ops(float* a, float b, float* out, size_t size);
  void smaller_equal_tensor_ops(float* a, float* b, float* out, size_t size);
  void smaller_equal_scalar_ops(float* a, float b, float* out, size_t size);
  void transpose_1d_tensor_ops(float* a, float* out, int* shape);
  void transpose_2d_tensor_ops(float* a, float* out, int* shape);
  void transpose_3d_tensor_ops(float* a, float* out, int* shape);
  void transpose_ndim_tensor_ops(float* a, float* out, int* shape, int ndim);
  void compute_broadcast_indices(int linear_index, int* broadcasted_shape, int max_ndim,
      int a_ndim, int b_ndim, int* a_shape, int* b_shape, int* index_a, int* index_b);
}

#endif