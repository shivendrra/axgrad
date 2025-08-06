#include <stdio.h>
#include <stdlib.h>
#include "../cpu/ops_decomp.h"
#include "decompose.h"

Tensor** svd_tensor(Tensor* a) {
  if (a->ndim < 2) {
    fprintf(stderr, "Input array must be at least 2D for SVD\n");
    exit(EXIT_FAILURE);
  }

  int m = a->shape[a->ndim - 2], n = a->shape[a->ndim - 1], min_mn = (m < n) ? m : n;
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  size_t batch_size = 1;
  for (int i = 0; i < a->ndim - 2; i++) batch_size *= a->shape[i];
  size_t u_size = batch_size * m * m, s_size = batch_size * min_mn, vt_size = batch_size * n * n;  
  float *u = (float*)malloc(u_size * sizeof(float)), *s = (float*)malloc(s_size * sizeof(float)), *vt = (float*)malloc(vt_size * sizeof(float));
  if (a->ndim == 2) svd_ops(a_float, u, s, vt, a->shape);
  else batched_svd_ops(a_float, u, s, vt, a->shape, a->ndim);
  int *u_shape = (int*)malloc(a->ndim * sizeof(int)), *s_shape = (int*)malloc((a->ndim - 1) * sizeof(int)), *vt_shape = (int*)malloc(a->ndim * sizeof(int));
  for (int i = 0; i < a->ndim - 2; i++) {
    u_shape[i] = a->shape[i];
    s_shape[i] = a->shape[i];
    vt_shape[i] = a->shape[i];
  }
  u_shape[a->ndim - 2] = m; u_shape[a->ndim - 1] = m; s_shape[a->ndim - 2] = min_mn; vt_shape[a->ndim - 2] = n; vt_shape[a->ndim - 1] = n;
  Tensor* u_result = create_tensor(u, a->ndim, u_shape, u_size, a->dtype);
  Tensor* s_result = create_tensor(s, a->ndim - 1, s_shape, s_size, a->dtype);
  Tensor* vt_result = create_tensor(vt, a->ndim, vt_shape, vt_size, a->dtype);
  free(a_float); free(u); free(s); free(vt);
  free(u_shape); free(s_shape); free(vt_shape);
  Tensor** result = (Tensor**)malloc(3 * sizeof(Tensor*));
  result[0] = u_result; result[1] = s_result; result[2] = vt_result;
  free(u_result); free(s_result); free(vt_result);
  return result;
}

Tensor* cholesky_tensor(Tensor* a) {
  if (a->ndim < 2) {
    fprintf(stderr, "Input array must be at least 2D for Cholesky decomposition\n");
    exit(EXIT_FAILURE);
  }
  int last_dim = a->shape[a->ndim - 1];
  int second_last_dim = a->shape[a->ndim - 2];
  if (last_dim != second_last_dim) {
    fprintf(stderr, "Matrix must be square for Cholesky decomposition: %d != %d\n", second_last_dim, last_dim);
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  int* result_shape = (int*)malloc(a->ndim * sizeof(int));
  for (size_t i = 0; i < a->ndim; i++) result_shape[i] = a->shape[i];
  size_t result_size = a->size;
  float* out = (float*)malloc(result_size * sizeof(float));
  if (a->ndim == 2) chol_ops(a_float, out, a->shape);
  else batched_chol_ops(a_float, out, a->shape, a->ndim);
  Tensor* result = create_tensor(out, a->ndim, result_shape, result_size, a->dtype);
  free(a_float); free(out); free(result_shape);
  return result;
}

Tensor* eig_tensor(Tensor* a) {
  if (a->ndim != 2) {
    fprintf(stderr, "Only 2D array supported for eig()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[0] != a->shape[1]) {
    fprintf(stderr, "Tensor must be square to compute eigenvalues. dim0 '%d' != dim1 '%d'\n", a->shape[0], a->shape[1]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(1 * sizeof(int));
  shape[0] = a->shape[0]; // eigenvalues count equals matrix dimension
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->shape[0] * sizeof(float));
  eigenvals_ops_tensor(a_float, out, a->shape[0]);
  Tensor* result = create_tensor(out, 1, shape, a->shape[0], a->dtype);
  free(a_float); free(out); free(shape);
  return result;
}

Tensor* eigv_tensor(Tensor* a) {
  if (a->ndim != 2) {
    fprintf(stderr, "Only 2D array supported for eigv()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[0] != a->shape[1]) {
    fprintf(stderr, "Tensor must be square to compute eigenvectors. dim0 '%d' != dim1 '%d'\n", a->shape[0], a->shape[1]);
    exit(EXIT_FAILURE);
  }
  int* shape = (int*)malloc(2 * sizeof(int));
  shape[0] = a->shape[0]; shape[1] = a->shape[1]; // same dimensions as input matrix
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->size * sizeof(float));
  eigenvecs_ops_tensor(a_float, out, a->shape[0]);
  Tensor* result = create_tensor(out, 2, shape, a->size, a->dtype);
  free(a_float); free(out); free(shape);
  return result;
}

Tensor* eigh_tensor(Tensor* a) {
  if (a->ndim != 2) {
    fprintf(stderr, "Only 2D array supported for eigh()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[0] != a->shape[1]) {
    fprintf(stderr, "Tensor must be square to compute hermitian eigenvalues. dim0 '%d' != dim1 '%d'\n", a->shape[0], a->shape[1]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(1 * sizeof(int));
  shape[0] = a->shape[0]; // eigenvalues count equals matrix dimension
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->shape[0] * sizeof(float));
  eigenvals_h_ops_tensor(a_float, out, a->shape[0]);
  Tensor* result = create_tensor(out, 1, shape, a->shape[0], a->dtype);
  free(a_float); free(out); free(shape);
  return result;
}

Tensor* eighv_tensor(Tensor* a) {
  if (a->ndim != 2) {
    fprintf(stderr, "Only 2D array supported for eighv()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[0] != a->shape[1]) {
    fprintf(stderr, "Tensor must be square to compute hermitian eigenvectors. dim0 '%d' != dim1 '%d'\n", a->shape[0], a->shape[1]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(2 * sizeof(int));
  shape[0] = a->shape[0]; shape[1] = a->shape[1]; // same dimensions as input matrix
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->size * sizeof(float));
  eigenvecs_h_ops_tensor(a_float, out, a->shape[0]);
  Tensor* result = create_tensor(out, 2, shape, a->size, a->dtype);
  free(a_float); free(out); free(shape);
  return result;
}

Tensor* batched_eig_tensor(Tensor* a) {
  if (a->ndim != 3) {
    fprintf(stderr, "Only 3D array supported for batched eig()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[1] != a->shape[2]) {
    fprintf(stderr, "Tensor must be square to compute eigenvalues. dim1 '%d' != dim2 '%d'\n", a->shape[1], a->shape[2]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(2 * sizeof(int));
  shape[0] = a->shape[0], shape[1] = a->shape[1];
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->shape[0] * a->shape[1] * sizeof(float));
  batched_eigenvals_ops(a_float, out, a->shape[1], a->shape[0]);
  Tensor* result = create_tensor(out, 2, shape, a->shape[0] * a->shape[1], a->dtype);
  free(a_float); free(out); free(shape);
  return result;
}

Tensor* batched_eigv_tensor(Tensor* a) {
  if (a->ndim != 3) {
    fprintf(stderr, "Only 3D array supported for batched eigv()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[1] != a->shape[2]) {
    fprintf(stderr, "Tensor must be square to compute eigenvectors. dim1 '%d' != dim2 '%d'\n", a->shape[1], a->shape[2]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(3 * sizeof(int));
  shape[0] = a->shape[0], shape[1] = a->shape[1], shape[2] = a->shape[2];
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->size * sizeof(float));
  batched_eigenvecs_ops(a_float, out, a->shape[1], a->shape[0]);
  Tensor* result = create_tensor(out, 3, shape, a->size, a->dtype);
  free(a_float); free(out); free(shape);
  return result;
}

Tensor* batched_eigh_tensor(Tensor* a) {
  if (a->ndim != 3) {
    fprintf(stderr, "Only 3D array supported for batched eigh()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[1] != a->shape[2]) {
    fprintf(stderr, "Tensor must be square to compute hermitian eigenvalues. dim1 '%d' != dim2 '%d'\n", a->shape[1], a->shape[2]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(2 * sizeof(int));
  shape[0] = a->shape[0], shape[1] = a->shape[1];
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->shape[0] * a->shape[1] * sizeof(float));
  batched_eigenvals_h_ops(a_float, out, a->shape[1], a->shape[0]);
  Tensor* result = create_tensor(out, 2, shape, a->shape[0] * a->shape[1], a->dtype);
  free(a_float); free(out); free(shape);
  return result;
}

Tensor* batched_eighv_tensor(Tensor* a) {
  if (a->ndim != 3) {
    fprintf(stderr, "Only 3D array supported for batched eighv()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[1] != a->shape[2]) {
    fprintf(stderr, "Tensor must be square to compute hermitian eigenvectors. dim1 '%d' != dim2 '%d'\n", a->shape[1], a->shape[2]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(3 * sizeof(int));
  shape[0] = a->shape[0], shape[1] = a->shape[1], shape[2] = a->shape[2];
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->size * sizeof(float));
  batched_eigenvecs_h_ops(a_float, out, a->shape[1], a->shape[0]);
  Tensor* result = create_tensor(out, 3, shape, a->size, a->dtype);
  free(a_float); free(out); free(shape);
  return result;
}

Tensor** qr_tensor(Tensor* a) {
  if (a->ndim != 2) {
    fprintf(stderr, "Only 2D array supported for qr()\n");
    exit(EXIT_FAILURE);
  }

  int m = a->shape[0], n = a->shape[1];
  int *q_shape = (int*)malloc(2 * sizeof(int)), *r_shape = (int*)malloc(2 * sizeof(int));
  q_shape[0] = m; q_shape[1] = m; r_shape[0] = m; r_shape[1] = n;
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *q_out = (float*)malloc(m * m * sizeof(float)), *r_out = (float*)malloc(m * n * sizeof(float));
  qr_decomp_ops(a_float, q_out, r_out, a->shape);
  Tensor** result = (Tensor**)malloc(2 * sizeof(Tensor*));
  result[0] = create_tensor(q_out, 2, q_shape, m * m, a->dtype);
  result[1] = create_tensor(r_out, 2, r_shape, m * n, a->dtype);
  free(a_float); free(q_out); free(r_out); free(q_shape); free(r_shape);
  return result;
}

Tensor** batched_qr_tensor(Tensor* a) {
  if (a->ndim < 2) {
    fprintf(stderr, "Tensor must have at least 2 dimensions for batched qr()\n");
    exit(EXIT_FAILURE);
  }

  int m = a->shape[a->ndim - 2], n = a->shape[a->ndim - 1], batch_size = 1;
  for (int i = 0; i < a->ndim - 2; i++) { batch_size *= a->shape[i]; }
  int *q_shape = (int*)malloc(a->ndim * sizeof(int)), *r_shape = (int*)malloc(a->ndim * sizeof(int));
  for (int i = 0; i < a->ndim - 2; i++) {
    q_shape[i] = a->shape[i];
    r_shape[i] = a->shape[i];
  }
  q_shape[a->ndim - 2] = m; q_shape[a->ndim - 1] = m;
  r_shape[a->ndim - 2] = m; r_shape[a->ndim - 1] = n;
  float* a_float = convert_to_float32(a->data, a->dtype, a->size), *q_out = (float*)malloc(batch_size * m * m * sizeof(float)), *r_out = (float*)malloc(batch_size * m * n * sizeof(float));
  batched_qr_decomp_ops(a_float, q_out, r_out, a->shape, a->ndim);
  Tensor** result = (Tensor**)malloc(2 * sizeof(Tensor*));
  result[0] = create_tensor(q_out, a->ndim, q_shape, batch_size * m * m, a->dtype);
  result[1] = create_tensor(r_out, a->ndim, r_shape, batch_size * m * n, a->dtype);
  free(a_float); free(q_out); free(r_out); free(q_shape); free(r_shape);
  return result;
}

Tensor** lu_tensor(Tensor* a) {
  if (a->ndim != 2) {
    fprintf(stderr, "Only 2D array supported for lu()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[0] != a->shape[1]) {
    fprintf(stderr, "Tensor must be square for lu(). dim0 '%d' != dim1 '%d'\n", a->shape[0], a->shape[1]);
    exit(EXIT_FAILURE);
  }

  int n = a->shape[0];
  int *l_shape = (int*)malloc(2 * sizeof(int)), *u_shape = (int*)malloc(2 * sizeof(int));
  l_shape[0] = n; l_shape[1] = n;
  u_shape[0] = n; u_shape[1] = n;
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *l_out = (float*)malloc(n * n * sizeof(float)), *u_out = (float*)malloc(n * n * sizeof(float));
  int* p_out = (int*)malloc(n * sizeof(int));
  lu_decomp_ops(a_float, l_out, u_out, p_out, a->shape);
  Tensor** result = (Tensor**)malloc(2 * sizeof(Tensor*));
  result[0] = create_tensor(l_out, 2, l_shape, n * n, a->dtype);
  result[1] = create_tensor(u_out, 2, u_shape, n * n, a->dtype);
  free(a_float); free(l_out); free(u_out); free(p_out); free(l_shape); free(u_shape); 
  return result;
}

Tensor** batched_lu_tensor(Tensor* a) {
  if (a->ndim < 2) {
    fprintf(stderr, "Tensor must have at least 2 dimensions for batched lu()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[a->ndim - 1] != a->shape[a->ndim - 2]) {
    fprintf(stderr, "Tensor must be square for lu(). dim%d '%d' != dim%d '%d'\n", a->ndim - 2, a->shape[a->ndim - 2], a->ndim - 1, a->shape[a->ndim - 1]);
    exit(EXIT_FAILURE);
  }

  int n = a->shape[a->ndim - 1];
  int batch_size = 1;
  for (int i = 0; i < a->ndim - 2; i++) batch_size *= a->shape[i];
  int *l_shape = (int*)malloc(a->ndim * sizeof(int)), *u_shape = (int*)malloc(a->ndim * sizeof(int));
  for (int i = 0; i < a->ndim; i++) {
    l_shape[i] = a->shape[i];
    u_shape[i] = a->shape[i];
  }
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *l_out = (float*)malloc(a->size * sizeof(float)), *u_out = (float*)malloc(a->size * sizeof(float));
  int* p_out = (int*)malloc(batch_size * n * sizeof(int));
  batched_lu_decomp_ops(a_float, l_out, u_out, p_out, a->shape, a->ndim);
  Tensor** result = (Tensor**)malloc(2 * sizeof(Tensor*));
  result[0] = create_tensor(l_out, a->ndim, l_shape, a->size, a->dtype);
  result[1] = create_tensor(u_out, a->ndim, u_shape, a->size, a->dtype);
  free(a_float); free(l_out); free(u_out); free(p_out); free(l_shape); free(u_shape);
  return result;
}