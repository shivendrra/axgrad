#ifndef __OPS_DECOMP__H__
#define __OPS_DECOMP__H__

#include <stddef.h>

extern "C" {
  void svd_ops(float* a, float* u, float* s, float* vt, int* shape);
  void batched_svd_ops(float* a, float* u, float* s, float* vt, int* shape, int ndim);
  void chol_ops(float* a, float* l, int* shape);
  void batched_chol_ops(float* a, float* l, int* shape, int ndim);
  void qr_decomp_ops(float* a, float* q, float* r, int* shape);
  void batched_qr_decomp_ops(float* a, float* q, float* r, int* shape, int ndim);
  void lu_decomp_ops(float* a, float* l, float* u, int* p, int* shape);
  void batched_lu_decomp_ops(float* a, float* l, float* u, int* p, int* shape, int ndim);
  void eigenvals_ops_tensor(float* a, float* eigenvals, size_t size);
  void batched_eigenvals_ops(float* a, float* eigenvals, size_t size, size_t batch);
  void eigenvecs_ops_tensor(float* a, float* eigenvecs, size_t size);
  void batched_eigenvecs_ops(float* a, float* eigenvecs, size_t size, size_t batch);
  void eigenvals_h_ops_tensor(float* a, float* eigenvals, size_t size);
  void batched_eigenvals_h_ops(float* a, float* eigenvals, size_t size, size_t batch);
  void eigenvecs_h_ops_tensor(float* a, float* eigenvecs, size_t size);
  void batched_eigenvecs_h_ops(float* a, float* eigenvecs, size_t size, size_t batch);
}

#endif  //!__OPS_DECOMP__H__