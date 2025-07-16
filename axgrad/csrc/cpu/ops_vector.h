#ifndef __OPS_VECTOR__H__
#define __OPS_VECTOR__H__

#include <stddef.h>

extern "C" {
  void vector_dot_ops(float* a, float* b, float* out, size_t size);
  void vector_matrix_dot_ops(float* vec, float* mat, float* out, size_t size_v, size_t size_m);
  void vector_outer_product_ops(float* a, float* b, float* out, size_t size_n, size_t size_m);
  void vector_inner_product_ops(float* a, float* b, float* out, size_t size);
  void cross_1d_ops(float* a, float* b, float* out, size_t size);
  void cross_2d_ops(float* a, float* b, float* out, size_t rows, size_t cols, size_t axis);
  void cross_3d_ops(float* a, float* b, float* out, size_t dim0, size_t dim1, size_t dim2, size_t axis);
}

#endif  //!__OPS_VECTOR__H__