#ifndef __OPS_MATRIX__H__
#define __OPS_MATRIX__H__

#include <stddef.h>

extern "C" {
  void det_ops_tensor(float* a, float* out, size_t size);
  void batched_det_ops(float* a, float* out, size_t size, size_t batch);
  void inv_ops(float* a, float* out, int* shape);
  void batched_inv_ops(float* a, float* out, int* shape, int ndim);
  void solve_ops(float* a, float* b, float* out, int* shape_a, int* shape_b);
  void batched_solve_ops(float* a, float* b, float* out, int* shape_a, int* shape_b, int ndim);
  void lstsq_ops(float* a, float* b, float* out, int* shape_a, int* shape_b);
  void batched_lstsq_ops(float* a, float* b, float* out, int* shape_a, int* shape_b, int ndim);
}

#endif