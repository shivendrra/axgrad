#include <stdlib.h>
#include <stddef.h>
#include "ops_shape.h"

void reassign_tensor_ops(float* a, float* out, size_t size) {
  for (int i = 0; i < size; i++) { out[i] = a[i]; }
}

void equal_tensor_ops(float* a, float* b, float* out, size_t size) {
  for (int i = 0; i < size; i++) { out[i] = (a[i] == b[i]) ? 1 : 0;}
}

void transpose_1d_tensor_ops(float* a, float* out, int* shape) {
  for (int i = 0; i < shape[0]; i++) { out[i] = a[i]; }
}

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