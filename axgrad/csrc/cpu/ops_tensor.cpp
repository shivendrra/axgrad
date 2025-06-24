#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include "ops_tensor.h"
#include "ops_shape.h"

// Optimized matrix multiplication using transposed second matrix
// A: shape_a[0] x shape_a[1], B^T: shape_b[1] x shape_b[0], C: shape_a[0] x shape_b[0]
// This computes C = A @ B where B is provided in transposed form
void matmul_tensor_ops(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int rows_a = shape_a[0];    // rows in 'a'
  int cols_a = shape_a[1];    // cols in 'a' 
  int rows_b = shape_b[0];    // rows in 'b' (original 'b' before transpose)
  int cols_b = shape_b[1];    // cols in 'b' (original 'b' before transpose)
  float* b_transposed = (float*)malloc(rows_b * cols_b * sizeof(float));
  if (b_transposed == NULL) {
    fprintf(stderr, "Memory allocation failed for transpose buffer\n");
    exit(EXIT_FAILURE);
  }
  transpose_2d_tensor_ops(b, b_transposed, shape_b);
  // for a @ b^T: a(rows_a x cols_a) @ b^T(cols_b x rows_b) = out(rows_a x rows_b)
  // we need cols_a == cols_b for this to work
  for (int i = 0; i < rows_a; i++) {
    for (int j = 0; j < cols_b; j++) {
      float sum = 0.0f;
      // dot product between row i of A and row j of B^T (which is column j of original B)
      for (int k = 0; k < cols_a; k++) {
        sum += a[i * cols_a + k] * b_transposed[j * cols_a + k];
      }
      out[i * cols_b + j] = sum;
    }
  }
}

// batch matrix multiplication: batched A @ batched B
// A: shape1[0] x shape1[1] x shape1[2], B: shape2[0] x shape2[1] x shape2[2]
// output: shape1[0] x shape1[1] x shape2[2] (assuming shape1[0] == shape2[0])
void batch_matmul_tensor_ops(float* a, float* b, float* out, int* shape1, int* shape2, int* strides1, int* strides2) {
  int batch_size = shape1[0];
  int out_stride = shape1[1] * shape2[2];
  
  for (int batch = 0; batch < batch_size; batch++) {
    for (int i = 0; i < shape1[1]; i++) {
      for (int j = 0; j < shape2[2]; j++) {
        float sum = 0.0f;
        for (int k = 0; k < shape1[2]; k++) {
          float a_val = a[batch * strides1[0] + i * shape1[2] + k];
          float b_val = b[batch * strides2[0] + k * shape2[2] + j];
          sum += a_val * b_val;
        }
        out[batch * out_stride + i * shape2[2] + j] = sum;
      }
    }
  }
}

// broadcasted matrix multiplication: single A * batched B
// A: shape1[0] x shape1[1], B: shape2[0] x shape2[1] x shape2[2]
// output: shape2[0] x shape1[0] x shape2[2]
void broadcasted_matmul_tensor_ops(float* a, float* b, float* out, int* shape1, int* shape2, int* strides1, int* strides2) {
  int out_stride = shape1[0] * shape2[2];
  
  for (int batch = 0; batch < shape2[0]; batch++) {
    for (int i = 0; i < shape1[0]; i++) {
      for (int j = 0; j < shape2[2]; j++) {
        float sum = 0.0f;
        for (int k = 0; k < shape1[1]; k++) {
          // A is broadcasted across batches, B is batched
          float a_val = a[i * shape1[1] + k];
          float b_val = b[batch * strides2[0] + k * shape2[2] + j];
          sum += a_val * b_val;
        }
        out[batch * out_stride + i * shape2[2] + j] = sum;
      }
    }
  }
}

// Dot product of two 1D vectors
// computes sum(a[i] * b[i]) for i = 0 to size-1
void dot_tensor_ops(float* a, float* b, float* out, size_t size) {
  float sum = 0.0f;
  for (size_t i = 0; i < size; i++) {
    sum += a[i] * b[i];
  }
  *out = sum;
}

// batch dot product of multiple pairs of 1D vectors
// a: batch_count x vector_size (flattened), b: batch_count x vector_size (flattened)
// out: batch_count (output tensor of dot products)
void batch_dot_tensor_ops(float* a, float* b, float* out, size_t batch_count, size_t vector_size) {
  for (size_t batch = 0; batch < batch_count; batch++) {
    float sum = 0.0f;
    size_t batch_offset = batch * vector_size;
    
    for (size_t i = 0; i < vector_size; i++) {
      sum += a[batch_offset + i] * b[batch_offset + i];
    }
    out[batch] = sum;
  }
}