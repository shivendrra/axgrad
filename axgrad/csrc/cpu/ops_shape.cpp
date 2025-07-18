#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include "ops_shape.h"

void reassign_tensor_ops(float* a, float* out, size_t size) { for (int i = 0; i < size; i++) { out[i] = a[i]; } }
void equal_tensor_ops(float* a, float* b, float* out, size_t size) { for (int i = 0; i < size; i++) { out[i] = (a[i] == b[i]) ? 1 : 0;} }
void equal_scalar_ops(float* a, float b, float* out, size_t size) { for (int i = 0; i < size; i++) { out[i] = (a[i] == b) ? 1 : 0;} }
void not_equal_tensor_ops(float* a, float* b, float* out, size_t size) { for (int i = 0; i < size; i++) { out[i] = (a[i] != b[i]) ? 1 : 0;} }
void not_equal_scalar_ops(float* a, float b, float* out, size_t size) { for (int i = 0; i < size; i++) { out[i] = (a[i] != b) ? 1 : 0;} }
void greater_tensor_ops(float* a, float* b, float* out, size_t size) { for (int i = 0; i < size; i++) { out[i] = (a[i] > b[i]) ? 1: 0;} }
void greater_scalar_ops(float* a, float b, float* out, size_t size) { for (int i = 0; i < size; i++) { out[i] = (a[i] > b) ? 1 : 0;} }
void greater_equal_tensor_ops(float* a, float* b, float* out, size_t size) { for (int i = 0; i < size; i++) { out[i] = (a[i] >= b[i]) ? 1: 0;} }
void greater_equal_scalar_ops(float* a, float b, float* out, size_t size) { for (int i = 0; i < size; i++) { out[i] = (a[i] >= b) ? 1 : 0;} }
void smaller_tensor_ops(float* a, float* b, float* out, size_t size) { for (int i = 0; i < size; i++) { out[i] = (a[i] < b[i]) ? 1: 0;} }
void smaller_scalar_ops(float* a, float b, float* out, size_t size) { for (int i = 0; i < size; i++) { out[i] = (a[i] < b) ? 1: 0;} }
void smaller_equal_tensor_ops(float* a, float* b, float* out, size_t size) { for (int i = 0; i < size; i++) { out[i] = (a[i] <= b[i]) ? 1: 0;} }
void smaller_equal_scalar_ops(float* a, float b, float* out, size_t size) { for (int i = 0; i < size; i++) { out[i] = (a[i] <= b) ? 1: 0;} }
void transpose_1d_tensor_ops(float* a, float* out, int* shape) { for (int i = 0; i < shape[0]; i++) { out[i] = a[i]; } }

void transpose_2d_tensor_ops(float* a, float* out, int* shape) {
  int rows = shape[0], cols = shape[1];
  for (int idx = 0; idx < rows * cols ; ++idx) {
    // Transpose: out[j][i] = a[i][j]
    int i = idx / cols, j = idx % cols;
    // out is cols x rows, so out[j][i] = out[j * rows + i]
    // a is rows x cols, so a[i][j] = a[idx]
    // this brings down time complexity from O(n2) -> O(n)
    out[j * rows + i] = a[idx];
  }
}

void transpose_3d_tensor_ops(float* a, float* out, int* shape) {
  int B = shape[0], R = shape[1], C = shape[2];
  int total = B * R * C;
  
  // For 3D transpose, we reverse all dimensions: (d0, d1, d2) -> (d2, d1, d0)
  for (int idx = 0; idx < total; ++idx) {
    int b = idx / (R * C), rem = idx % (R * C);
    int i = rem / C, j = rem % C;

    // Original: a[i][j][k] = a[i * dim1 * dim2 + j * dim2 + k]
    // Transposed: out[k][j][i] = out[k * dim1 * dim0 + j * dim0 + i]
    // this brings down time complexity from O(n3) -> O(n)
    int out_idx = b * (C * R) + j * R + i;
    out[out_idx] = a[idx];
  }
}

void transpose_ndim_tensor_ops(float* a, float* out, int* shape, int ndim) {
  // calculating total size for verification
  size_t total_size = 1;
  for (int i = 0; i < ndim; i++) {
    total_size *= shape[i];
  }
  // create transposed shape for coordinate calculations
  int transposed_shape[ndim];
  for (int i = 0; i < ndim; i++) {
    transposed_shape[i] = shape[ndim - 1 - i];
  }

  // for each element in the output tensor
  for (size_t out_idx = 0; out_idx < total_size; out_idx++) {
    // converting flat output index to multi-dimensional coordinates
    size_t temp = out_idx;
    int out_coords[ndim];
    
    // calculating coordinates in transposed space using transposed_shape
    for (int i = ndim - 1; i >= 0; i--) {
      out_coords[i] = temp % transposed_shape[i];
      temp /= transposed_shape[i];
    }

    // converting to input coordinates (reverse the coordinate order)
    int in_coords[ndim];
    for (int i = 0; i < ndim; i++) {
      in_coords[i] = out_coords[ndim - 1 - i];
    }

    // calculating input flat index from coordinates using original shape
    size_t in_idx = 0;
    size_t multiplier = 1;
    for (int i = ndim - 1; i >= 0; i--) {
      in_idx += in_coords[i] * multiplier;
      multiplier *= shape[i];
    }
    out[out_idx] = a[in_idx];     // copying the element
  }
}

void compute_broadcast_indices(int linear_index, int* broadcasted_shape, int max_ndim,  int a_ndim, int b_ndim, int* a_shape, int* b_shape, int* index_a, int* index_b) {
  int *strides_a = (int*)malloc(max_ndim * sizeof(int)), *strides_b = (int*)malloc(max_ndim * sizeof(int));
  if (strides_a == NULL || strides_b == NULL) {
    fprintf(stderr, "Couldn't assign the strides to memory, operation failed!\n");
    exit(1);
  }

  int stride_a = 1, stride_b = 1;
  for (int i = max_ndim - 1; i >= 0; i--) {
    int dim_a = (i >= max_ndim - a_ndim) ? a_shape[i - (max_ndim - a_ndim)] : 1, dim_b = (i >= max_ndim - b_ndim) ? b_shape[i - (max_ndim - b_ndim)] : 1;
    strides_a[i] = (dim_a == broadcasted_shape[i]) ? stride_a : 0, strides_b[i] = (dim_b == broadcasted_shape[i]) ? stride_b : 0;
    stride_a *= dim_a, stride_b *= dim_b;
  }
  *index_a = 0; *index_b = 0;
  int temp_index = linear_index;
  for (int j = max_ndim - 1; j >= 0; j--) {
    int pos = temp_index % broadcasted_shape[j];
    temp_index /= broadcasted_shape[j];
    if (strides_a[j] != 0) *index_a += pos * strides_a[j];
    if (strides_b[j] != 0) *index_b += pos * strides_b[j];
  }
  free(strides_a);
  free(strides_b);
}