#include <stdio.h>
#include <stdlib.h>
#include "../cpu/ops_matrix.h"
#include "matrix.h"

Tensor* det_tensor(Tensor* a) {
  if (a->ndim != 2) {
    fprintf(stderr, "Only 2D array supported for det()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[0] != a->shape[1]) {
    fprintf(stderr, "Tensor must be square to compute det(). dim0 '%d' != dim1 '%d'\n", a->shape[0], a->shape[1]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(1 * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  shape[0] = 1;
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* out = (float*)malloc(1 * sizeof(float));
  // Passing matrix dimension (shape[0]), not total size
  det_ops_tensor(a_float, out, a->shape[0]);
  Tensor* result = create_tensor(out, 1, shape, 1, a->dtype);
  free(a_float); free(out); free(shape);
  return result;
}

Tensor* batched_det_tensor(Tensor* a) {
  if (a->ndim != 3) {
    fprintf(stderr, "Only 3D array supported for batched det()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[1] != a->shape[2]) {
    fprintf(stderr, "Tensor must be square to compute det(). dim1 '%d' != dim2 '%d'\n", a->shape[1], a->shape[2]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(1 * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  shape[0] = a->shape[0]; // Output should have batch size
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* out = (float*)malloc(a->shape[0] * sizeof(float)); // allocating for batch size
  // Pass matrix dimension (shape[1])
  batched_det_ops(a_float, out, a->shape[1], a->shape[0]);
  Tensor* result = create_tensor(out, 1, shape, a->shape[0], a->dtype);
  free(a_float); free(out); free(shape);
  return result;
}

Tensor* inv_tensor(Tensor* a) {
  if (a->ndim < 2) {
    fprintf(stderr, "Input array must be at least 2D for matrix inverse\n");
    exit(EXIT_FAILURE);
  }

  int last_dim = a->shape[a->ndim - 1], second_last_dim = a->shape[a->ndim - 2];
  if (last_dim != second_last_dim) {
    fprintf(stderr, "Matrix must be square for inverse: %d != %d\n", second_last_dim, last_dim);
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  int* result_shape = (int*)malloc(a->ndim * sizeof(int));

  for (size_t i = 0; i < a->ndim; i++) result_shape[i] = a->shape[i];
  size_t result_size = a->size;
  float* out = (float*)malloc(result_size * sizeof(float));
  if (a->ndim == 2) { inv_ops(a_float, out, a->shape); }
  else { batched_inv_ops(a_float, out, a->shape, a->ndim); }
  Tensor* result = create_tensor(out, a->ndim, result_shape, result_size, a->dtype);
  free(a_float); free(out); free(result_shape);
  return result;
}

Tensor* solve_tensor(Tensor* a, Tensor* b) {
  if (a->ndim < 2 || b->ndim < 1) {
    fprintf(stderr, "Matrix 'a' must be at least 2D and vector 'b' must be at least 1D\n");
    exit(EXIT_FAILURE);
  }

  int a_rows = a->shape[a->ndim - 2], a_cols = a->shape[a->ndim - 1], b_rows = b->shape[b->ndim - 1];
  if (a_rows != a_cols) {
    fprintf(stderr, "Matrix 'a' must be square for solve: %d != %d\n", a_rows, a_cols);
    exit(EXIT_FAILURE);
  }
  if (a_rows != b_rows) {
    fprintf(stderr, "Matrix 'a' rows must match vector 'b' size: %d != %d\n", a_rows, b_rows);
    exit(EXIT_FAILURE);
  }

  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *b_float = convert_to_float32(b->data, b->dtype, b->size);
  int* result_shape = (int*)malloc(b->ndim * sizeof(int));

  for (size_t i = 0; i < b->ndim; i++) result_shape[i] = b->shape[i];
  size_t result_size = b->size;
  float* out = (float*)malloc(result_size * sizeof(float));
  if (a->ndim == 2 && b->ndim <= 2) {
    int shape_b[2] = {b->shape[b->ndim - 1], (b->ndim == 2) ? b->shape[1] : 1};
    solve_ops(a_float, b_float, out, a->shape + (a->ndim - 2), shape_b);
  } else {
    int shape_a_2d[2] = {a->shape[a->ndim - 2], a->shape[a->ndim - 1]};
    int shape_b_2d[2] = {b->shape[b->ndim - 1], (b->ndim >= 2) ? b->shape[b->ndim - 1] : 1};
    batched_solve_ops(a_float, b_float, out, shape_a_2d, shape_b_2d, a->ndim);
  }

  dtype_t result_dtype = promote_dtypes(a->dtype, b->dtype);
  Tensor* result = create_tensor(out, b->ndim, result_shape, result_size, result_dtype);
  free(a_float); free(b_float); free(out); free(result_shape);
  return result;
}

Tensor* lstsq_tensor(Tensor* a, Tensor* b) {
  if (a->ndim < 2 || b->ndim < 1) {
    fprintf(stderr, "Matrix 'a' must be at least 2D and vector 'b' must be at least 1D\n");
    exit(EXIT_FAILURE);
  }
  int a_rows = a->shape[a->ndim - 2], a_cols = a->shape[a->ndim - 1], b_rows = (b->ndim >= 2) ? b->shape[b->ndim - 2] : b->shape[0];
  if (a_rows != b_rows) {
    fprintf(stderr, "Matrix 'a' rows must match vector 'b' size: %d != %d\n", a_rows, b_rows);
    exit(EXIT_FAILURE);
  }

  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *b_float = convert_to_float32(b->data, b->dtype, b->size);
  size_t result_ndim;
  int* result_shape;  
  if (b->ndim == 1) {
    // b is 1D vector -> result is 1D with shape [a_cols]
    result_ndim = 1;
    result_shape = (int*)malloc(result_ndim * sizeof(int));
    result_shape[0] = a_cols;
  } else {
    // b is 2D or higher -> result keeps b's batch dimensions + [a_cols, b_cols]
    result_ndim = b->ndim;
    result_shape = (int*)malloc(result_ndim * sizeof(int));
    for (size_t i = 0; i < result_ndim - 2; i++) result_shape[i] = b->shape[i];
    if (result_ndim >= 2) {
      result_shape[result_ndim - 2] = a_cols;  // Number of unknowns
      result_shape[result_ndim - 1] = b->shape[b->ndim - 1];  // Number of RHS
    } else {
      result_shape[0] = a_cols;
    }
  }

  size_t result_size = 1;
  for (size_t i = 0; i < result_ndim; i++) result_size *= result_shape[i];
  float* out = (float*)malloc(result_size * sizeof(float));
  if (a->ndim == 2 && b->ndim <= 2) {
    int shape_a_2d[2] = {a_rows, a_cols};
    int shape_b_2d[2] = {b_rows, (b->ndim == 2) ? b->shape[1] : 1};
    lstsq_ops(a_float, b_float, out, shape_a_2d, shape_b_2d);
  } else {
    int shape_a_2d[2] = {a_rows, a_cols};
    int shape_b_2d[2] = {b_rows, (b->ndim >= 2) ? b->shape[b->ndim - 1] : 1};
    batched_lstsq_ops(a_float, b_float, out, shape_a_2d, shape_b_2d, a->ndim);
  }

  dtype_t result_dtype = promote_dtypes(a->dtype, b->dtype);
  Tensor* result = create_tensor(out, result_ndim, result_shape, result_size, result_dtype);
  free(a_float); free(b_float); free(out); free(result_shape);
  return result;
}