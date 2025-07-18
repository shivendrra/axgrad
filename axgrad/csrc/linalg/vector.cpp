#include <stdio.h>
#include <stdlib.h>
#include "../cpu/ops_vector.h"
#include "vector.h"

Tensor* vector_dot(Tensor* a, Tensor* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Tensors cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 1 || b->ndim != 1) {
    fprintf(stderr, "Only 1D tensors supported for dot product\n");
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
  if (a_float == NULL || b_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion");
    if (a_float) free(a_float);
    if (b_float) free(b_float);
    if (out) free(out);
    exit(EXIT_FAILURE);
  }

  vector_dot_ops(a_float, b_float, out, a->size);
  Tensor* result = create_tensor(out, 1, shape, 1, a->dtype);
  free(a_float); free(b_float); free(out); free(shape);
  return result;
}

Tensor* vector_matrix_dot(Tensor* vec, Tensor* mat) {
  if (vec == NULL || mat == NULL) {
    fprintf(stderr, "Tensors cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (vec->ndim != 1 || mat->ndim != 2) {
    fprintf(stderr, "Vector must be 1D and matrix must be 2D for vector-matrix dot product\n");
    exit(EXIT_FAILURE);
  }
  if (vec->shape[0] != mat->shape[1]) {
    fprintf(stderr, "Vector size must match matrix rows. vec_size '%d' != mat_rows '%d'\n", vec->shape[0], mat->shape[0]);
    exit(EXIT_FAILURE);
  }
  int* shape = (int*)malloc(1 * sizeof(int));
  shape[0] = mat->shape[1];
  float *vec_float = convert_to_float32(vec->data, vec->dtype, vec->size), *mat_float = convert_to_float32(mat->data, mat->dtype, mat->size);
  float* out = (float*)malloc(mat->shape[1] * sizeof(float));
  if (vec_float == NULL || mat_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion");
    if (vec_float) free(vec_float);
    if (mat_float) free(mat_float);
    if (out) free(out);
    exit(EXIT_FAILURE);
  }

  vector_matrix_dot_ops(vec_float, mat_float, out, vec->size, mat->size);
  Tensor* result = create_tensor(out, 1, shape, mat->shape[1], vec->dtype);
  free(vec_float); free(mat_float); free(out); free(shape);
  return result;
}

Tensor* vector_inner(Tensor* a, Tensor* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Tensors cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 1 || b->ndim != 1) {
    fprintf(stderr, "Only 1D tensors supported for inner product\n");
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
  if (a_float == NULL || b_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion");
    if (a_float) free(a_float);
    if (b_float) free(b_float);
    if (out) free(out);
    exit(EXIT_FAILURE);
  }

  vector_inner_product_ops(a_float, b_float, out, a->size);
  Tensor* result = create_tensor(out, 1, shape, 1, a->dtype);
  free(a_float); free(b_float); free(out); free(shape);
  return result;
}

Tensor* vector_outer(Tensor* a, Tensor* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Tensors cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 1 || b->ndim != 1) {
    fprintf(stderr, "Only 1D tensors supported for outer product\n");
    exit(EXIT_FAILURE);
  }
  int* shape = (int*)malloc(2 * sizeof(int));
  shape[0] = a->shape[0]; shape[1] = b->shape[0];
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *b_float = convert_to_float32(b->data, b->dtype, b->size);
  float* out = (float*)malloc(a->shape[0] * b->shape[0] * sizeof(float));
  if (a_float == NULL || b_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion");
    if (a_float) free(a_float);
    if (b_float) free(b_float);
    if (out) free(out);
    exit(EXIT_FAILURE);
  }

  vector_outer_product_ops(a_float, b_float, out, a->shape[0], b->shape[0]);
  Tensor* result = create_tensor(out, 2, shape, a->shape[0] * b->shape[0], a->dtype);
  free(a_float); free(b_float); free(out); free(shape);
  return result;
}

Tensor* vector_cross_axis(Tensor* a, Tensor* b, int axis) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Tensors cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "Tensors must have same number of dimensions for cross product\n");
    exit(EXIT_FAILURE);
  }

  // Validate axis bounds
  if (axis < 0) axis = a->ndim + axis;
  if (axis < 0 || axis >= a->ndim) {
    fprintf(stderr, "Axis %d is out of bounds for tensor of dimension %d\n", axis, a->ndim);
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
  if (a_float == NULL || b_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (b_float) free(b_float);
    if (out) free(out);
    if (out_shape) free(out_shape);
    exit(EXIT_FAILURE);
  }

  // calculating strides
  size_t* strides = (size_t*)malloc(a->ndim * sizeof(size_t));
  strides[a->ndim - 1] = 1;
  for (int i = a->ndim - 2; i >= 0; i--) { strides[i] = strides[i + 1] * a->shape[i + 1]; }
  if (a->ndim == 2) { cross_2d_ops(a_float, b_float, out, a->shape[0], a->shape[1], axis); }
  else if (a->ndim == 3) { cross_3d_ops(a_float, b_float, out, a->shape[0], a->shape[1], a->shape[2], axis); }
  else {
    fprintf(stderr, "Only 2D and 3D tensors supported for cross product with axis\n");
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
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Tensors cannot be null!\n");
    exit(EXIT_FAILURE);
  }
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
    fprintf(stderr, "Only 1D, 2D, and 3D tensors supported for cross product\n");
    exit(EXIT_FAILURE);
  }

  Tensor* result = create_tensor(out, a->ndim, shape, a->size, a->dtype);
  free(a_float); free(b_float); free(out); free(shape);
  return result;
}