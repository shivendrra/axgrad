#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include "ops_vector.h"
#include "ops_tensor.h"

void vector_dot_ops(float* a, float* b, float* out, size_t size) { dot_tensor_ops(a, b, out, size); }
void vector_inner_product_ops(float* a, float* b, float* out, size_t size) { dot_tensor_ops(a, b, out, size); }

// Vector-matrix multiplication: out = vec * mat
// vec is 1 x size_n, mat is size_n x size_m, out is 1 x size_m
void vector_matrix_dot_ops(float* vec, float* mat, float* out, size_t size_v, size_t size_m) {
  size_t cols = size_m / size_v;
  for (size_t j = 0; j < cols; j++) {
    out[j] = 0.0f;
    for (size_t i = 0; i < size_v; i++) {
      out[j] += vec[i] * mat[i * cols + j];   // M stored in row-major order
    }
  }
}

// Vector outer product: out = a @ b
// a is nx1, b is mx1, out is nxm matrix
void vector_outer_product_ops(float* a, float* b, float* out, size_t size_n, size_t size_m) {
  for (size_t i = 0; i < size_n; i++) {
    for (size_t j = 0; j < size_m; j++) {
      out[i * size_m + j] = a[i] * b[j];  // out stored in row-major order
    }
  }
}

void cross_product_ops(float* a, float* b, float* out, size_t* shape, size_t ndim, size_t axis, size_t* a_stride, size_t* b_stride) {
  size_t axis_size = shape[axis];
  // Calculate total number of elements excluding the axis dimension
  size_t total_elements = 1;
  for (size_t i = 0; i < ndim; i++) { if (i != axis) { total_elements *= shape[i]; } }
  if (axis_size == 2) {
    // 2D cross product: returns scalar for each cross product
    for (size_t i = 0; i < total_elements; i++) {
      // Convert flat index to multi-dimensional coordinates
      size_t temp_i = i, a_idx = 0, b_idx = 0;
      for (size_t dim = 0; dim < ndim; dim++) {
        if (dim != axis) {
          size_t dim_size = shape[dim];
          size_t coord = temp_i % dim_size;
          temp_i /= dim_size;
          a_idx += coord * a_stride[dim];
          b_idx += coord * b_stride[dim];
        }
      }
      float a0 = a[a_idx], a1 = a[a_idx + a_stride[axis]];
      float b0 = b[b_idx], b1 = b[b_idx + b_stride[axis]];
      out[i] = a0 * b1 - a1 * b0;
    }
  }
  else if (axis_size == 3) {
    // 3D cross product: returns 3-element vector for each cross product
    for (size_t i = 0; i < total_elements; i++) {
      size_t temp_i = i, a_idx = 0, b_idx = 0;
      for (size_t dim = 0; dim < ndim; dim++) {
        if (dim != axis) {
          size_t dim_size = shape[dim];
          size_t coord = temp_i % dim_size;
          temp_i /= dim_size;
          a_idx += coord * a_stride[dim];
          b_idx += coord * b_stride[dim];
        }
      }

      float a0 = a[a_idx]; float a1 = a[a_idx + a_stride[axis]]; float a2 = a[a_idx + 2 * a_stride[axis]];
      float b0 = b[b_idx]; float b1 = b[b_idx + b_stride[axis]]; float b2 = b[b_idx + 2 * b_stride[axis]];
      out[i * 3 + 0] = a1 * b2 - a2 * b1;
      out[i * 3 + 1] = a2 * b0 - a0 * b2;
      out[i * 3 + 2] = a0 * b1 - a1 * b0;
    }
  }
}

void cross_1d_ops(float* a, float* b, float* out, size_t size) {
  if (size == 2) { out[0] = a[0] * b[1] - a[1] * b[0]; } // 2D cross product returns scalar
  else if (size == 3) { // 3D cross product returns vector
    out[0] = a[1] * b[2] - a[2] * b[1]; out[1] = a[2] * b[0] - a[0] * b[2]; out[2] = a[0] * b[1] - a[1] * b[0];
  }
}

// Cross product for 2D tensors (matrix of vectors)
void cross_2d_ops(float* a, float* b, float* out, size_t rows, size_t cols, size_t axis) {
  size_t shape[2] = {rows, cols};
  size_t stride[2] = {cols, 1};  // Row-major stride
  cross_product_ops(a, b, out, shape, 2, axis, stride, stride);
}

// Cross product for 3D tensors (tensor of vectors)
void cross_3d_ops(float* a, float* b, float* out, size_t dim0, size_t dim1, size_t dim2, size_t axis) {
  size_t shape[3] = {dim0, dim1, dim2};
  size_t stride[3] = {dim1 * dim2, dim2, 1};  // Row-major stride
  cross_product_ops(a, b, out, shape, 3, axis, stride, stride);
}