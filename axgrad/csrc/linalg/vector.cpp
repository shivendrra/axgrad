#include <stdio.h>
#include <stdlib.h>
#include "../cpu/ops_vector.h"
#include "vector.h"

Tensor* vector_dot(Tensor* a, Tensor* b) {
  if (a->ndim != 1 || b->ndim != 1) {
    fprintf(stderr, "Only 1D arrays supported for dot product\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[0] != b->shape[0]) {
    fprintf(stderr, "Tensors must have same size for dot product. size_a '%d' != size_b '%d'\n", a->shape[0], b->shape[0]);
    exit(EXIT_FAILURE);
  }
  int* shape = (int*)malloc(1 * sizeof(int));
  shape[0] = 1;
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *b_float = convert_to_float32(b->data, b->dtype, b->size);
  float* out = (float*)malloc(1 * sizeof(float));
  vector_dot_ops(a_float, b_float, out, a->size);
  Tensor* result = create_tensor(out, 1, shape, 1, a->dtype);
  free(a_float); free(b_float); free(out); free(shape);
  return result;
}

Tensor* vector_matrix_dot(Tensor* a, Tensor* b) {
  // Check if one is 1D and other is 2D
  if (!((a->ndim == 1 && b->ndim == 2) || (a->ndim == 2 && b->ndim == 1))) {
    fprintf(stderr, "One array must be 1D (vector) and other must be 2D (matrix)\n");
    exit(EXIT_FAILURE);
  }
  Tensor *vec, *mat;
  int is_matrix_vector = 0;  // 0 = vector-matrix, 1 = matrix-vector
  if (a->ndim == 1 && b->ndim == 2) {
    // vector-matrix: vec @ mat
    vec = a, mat = b, is_matrix_vector = 0;
    if (vec->shape[0] != mat->shape[0]) {
      fprintf(stderr, "Vector size must match matrix rows. vec_size '%d' != mat_rows '%d'\n", vec->shape[0], mat->shape[0]);
      exit(EXIT_FAILURE);
    }
  } else {
    // matrix-vector: mat @ vec
    mat = a, vec = b, is_matrix_vector = 1;
    if (mat->shape[1] != vec->shape[0]) {
      fprintf(stderr, "Matrix columns must match vector size. mat_cols '%d' != vec_size '%d'\n", mat->shape[1], vec->shape[0]);
      exit(EXIT_FAILURE);
    }
  }

  int* shape = (int*)malloc(1 * sizeof(int));
  int output_size;
  if (is_matrix_vector) {
    // matrix-vector: output size = number of matrix rows
    output_size = mat->shape[0];
    shape[0] = mat->shape[0];
  } else {
    // vector-matrix: output size = number of matrix columns
    output_size = mat->shape[1];
    shape[0] = mat->shape[1];
  }
  float *vec_float = convert_to_float32(vec->data, vec->dtype, vec->size), *mat_float = convert_to_float32(mat->data, mat->dtype, mat->size), * out = (float*)malloc(output_size * sizeof(float));
  if (is_matrix_vector) {
    matrix_vector_dot_ops(mat_float, vec_float, out, mat->size, vec->size);
  } else {
    vector_matrix_dot_ops(vec_float, mat_float, out, vec->size, mat->size);
  }
  dtype_t result_dtype = promote_dtypes(a->dtype, b->dtype);
  Tensor* result = create_tensor(out, 1, shape, output_size, result_dtype);
  free(vec_float); free(mat_float); free(out); free(shape);
  return result;
}

Tensor* vector_inner(Tensor* a, Tensor* b) {
  if (a->ndim != 1 || b->ndim != 1) {
    fprintf(stderr, "Only 1D arrays supported for inner product\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[0] != b->shape[0]) {
    fprintf(stderr, "Tensors must have same size for inner product. size_a '%d' != size_b '%d'\n", a->shape[0], b->shape[0]);
    exit(EXIT_FAILURE);
  }
  int* shape = (int*)malloc(1 * sizeof(int));
  shape[0] = 1;
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *b_float = convert_to_float32(b->data, b->dtype, b->size);
  float* out = (float*)malloc(1 * sizeof(float));
  vector_inner_product_ops(a_float, b_float, out, a->size);
  Tensor* result = create_tensor(out, 1, shape, 1, a->dtype);
  free(a_float); free(b_float); free(out); free(shape);
  return result;
}

Tensor* vector_outer(Tensor* a, Tensor* b) {
  if (a->ndim != 1 || b->ndim != 1) {
    fprintf(stderr, "Only 1D arrays supported for outer product\n");
    exit(EXIT_FAILURE);
  }
  int* shape = (int*)malloc(2 * sizeof(int));
  shape[0] = a->shape[0]; shape[1] = b->shape[0];
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *b_float = convert_to_float32(b->data, b->dtype, b->size);
  float* out = (float*)malloc(a->shape[0] * b->shape[0] * sizeof(float));
  vector_outer_product_ops(a_float, b_float, out, a->shape[0], b->shape[0]);
  Tensor* result = create_tensor(out, 2, shape, a->shape[0] * b->shape[0], a->dtype);
  free(a_float); free(b_float); free(out); free(shape);
  return result;
}

Tensor* vector_cross_axis(Tensor* a, Tensor* b, int axis) {
  if (a->ndim != b->ndim) {
    fprintf(stderr, "Tensors must have same number of dimensions for cross product\n");
    exit(EXIT_FAILURE);
  }
  // Validate axis bounds
  if (axis < 0) axis = a->ndim + axis;
  if (axis < 0 || axis >= a->ndim) {
    fprintf(stderr, "Axis %d is out of bounds for array of dimension %d\n", axis, a->ndim);
    exit(EXIT_FAILURE);
  }

  // Check if axis dimension is valid for cross product
  if (a->shape[axis] != 2 && a->shape[axis] != 3) {
    fprintf(stderr, "Cross product axis must have size 2 or 3, got %d\n", a->shape[axis]);
    exit(EXIT_FAILURE);
  }
  
  // Check shape compatibility
  for (int i = 0; i < a->ndim; i++) {
    if (a->shape[i] != b->shape[i]) {
      fprintf(stderr, "Tensors must have same shape for cross product\n");
      exit(EXIT_FAILURE);
    }
  }

  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *b_float = convert_to_float32(b->data, b->dtype, b->size);
  int* out_shape = (int*)malloc(a->ndim * sizeof(int));
  size_t out_size = 1;  
  for (int i = 0; i < a->ndim; i++) {
    if (i == axis) {
      if (a->shape[axis] == 2) { out_shape[i] = 1; } // 2D cross product reduces dimension - remove this axis
      else { out_shape[i] = 3; } // 3D cross product keeps size 3
    } else { out_shape[i] = a->shape[i]; }
    out_size *= out_shape[i]; }

  // For 2D cross product, we need to remove the axis dimension
  if (a->shape[axis] == 2) { out_size /= 1; }  // Adjust for the removed dimension
  float* out = (float*)malloc(out_size * sizeof(float));
  size_t* strides = (size_t*)malloc(a->ndim * sizeof(size_t));
  strides[a->ndim - 1] = 1;
  for (int i = a->ndim - 2; i >= 0; i--) { strides[i] = strides[i + 1] * a->shape[i + 1]; }
  if (a->ndim == 2) { cross_2d_ops(a_float, b_float, out, a->shape[0], a->shape[1], axis); }
  else if (a->ndim == 3) { cross_3d_ops(a_float, b_float, out, a->shape[0], a->shape[1], a->shape[2], axis); }
  else {
    fprintf(stderr, "Only 2D and 3D arrays supported for cross product with axis\n");
    free(a_float); free(b_float); free(out); free(out_shape); free(strides);
    exit(EXIT_FAILURE);
  }

  // handling output shape for 2D cross product (dimension reduction)
  if (a->shape[axis] == 2) {
    // creating new shape without the axis dimension
    int *final_shape = (int*)malloc((a->ndim - 1) * sizeof(int)), final_ndim = a->ndim - 1;
    int j = 0;
    for (int i = 0; i < a->ndim; i++) {
      if (i != axis) { final_shape[j++] = a->shape[i]; }
    }
    Tensor* result = create_tensor(out, final_ndim, final_shape, out_size, a->dtype);
    free(a_float); free(b_float); free(out); free(out_shape); free(strides); free(final_shape);
    return result;
  } else {
    Tensor* result = create_tensor(out, a->ndim, out_shape, out_size, a->dtype);
    free(a_float); free(b_float); free(out); free(strides);
    return result;
  }
}

Tensor* vector_cross(Tensor* a, Tensor* b) {
  if (a->ndim != b->ndim) {
    fprintf(stderr, "Tensors must have same number of dimensions for cross product\n");
    exit(EXIT_FAILURE);
  }
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *b_float = convert_to_float32(b->data, b->dtype, b->size);
  float* out = (float*)malloc(a->size * sizeof(float));
  if (a_float == NULL || b_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion");
    if (a_float) free(a_float);
    if (b_float) free(b_float);
    if (out) free(out);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(a->ndim * sizeof(int));
  for (int i = 0; i < a->ndim; i++) { shape[i] = a->shape[i]; }
  if (a->ndim == 1) {
    if (a->shape[0] != b->shape[0]) {
      fprintf(stderr, "Tensors must have same size for cross product. size_a '%d' != size_b '%d'\n", a->shape[0], b->shape[0]);
      exit(EXIT_FAILURE);
    }
    cross_1d_ops(a_float, b_float, out, a->size);
  } else if (a->ndim == 2) {
    if (a->shape[0] != b->shape[0] || a->shape[1] != b->shape[1]) {
      fprintf(stderr, "Tensors must have same shape for cross product\n");
      exit(EXIT_FAILURE);
    }
    cross_2d_ops(a_float, b_float, out, a->shape[0], a->shape[1], -1);
  } else if (a->ndim == 3) {
    if (a->shape[0] != b->shape[0] || a->shape[1] != b->shape[1] || a->shape[2] != b->shape[2]) {
      fprintf(stderr, "Tensors must have same shape for cross product\n");
      exit(EXIT_FAILURE);
    }
    cross_3d_ops(a_float, b_float, out, a->shape[0], a->shape[1], a->shape[2], -1);
  } else {
    fprintf(stderr, "Only 1D, 2D, and 3D arrays supported for cross product\n");
    exit(EXIT_FAILURE);
  }

  Tensor* result = create_tensor(out, a->ndim, shape, a->size, a->dtype);
  free(a_float); free(b_float); free(out); free(shape);
  return result;
}