#include <stdio.h>
#include <stdlib.h>
#include "transform.h"
#include "../core/dtype.h"
#include "../cpu/ops_tensor.h"
#include "../cpu/ops_shape.h"

// Helper function to add bias to matrix multiplication result
// Broadcasts bias vector across all rows of the result matrix
void add_bias_linear_ops(float* data, float* bias, int* shape, size_t ndim) {
  if (ndim == 1) {
    // 1D case: simple element-wise addition
    for (int i = 0; i < shape[0]; i++) { data[i] += bias[i]; }
  } else if (ndim == 2) {
    int rows = shape[0], cols = shape[1];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) { data[i * cols + j] += bias[j]; }
    }
  }
}

// Linear transformation for 1D input: y = Wx + b
// W: [out_features, in_features], x: [in_features], b: [out_features] (optional)
// Returns: [out_features]
Tensor* linear_1d_tensor(Tensor* weights, Tensor* input, Tensor* bias) {
  if (weights == NULL || input == NULL) {
    fprintf(stderr, "Weight and input tensors cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (weights->shape[1] != input->shape[0]) {
    fprintf(stderr, "Weight columns must match input size: %d != %d\n", weights->shape[1], input->shape[0]);
    exit(EXIT_FAILURE);
  }
  if (bias != NULL && (bias->ndim != 1 || bias->shape[0] != weights->shape[0])) {
    fprintf(stderr, "Bias must be 1D with size matching weight rows: %d\n", weights->shape[0]);
    exit(EXIT_FAILURE);
  }

  // Convert tensors to float32 for computation
  float* w_float = convert_to_float32(weights->data, weights->dtype, weights->size);
  float* x_float = convert_to_float32(input->data, input->dtype, input->size);
  float* b_float = NULL;
  if (bias != NULL) {
    b_float = convert_to_float32(bias->data, bias->dtype, bias->size);
  }

  if (w_float == NULL || x_float == NULL || (bias != NULL && b_float == NULL)) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (w_float) free(w_float);
    if (x_float) free(x_float);
    if (b_float) free(b_float);
    exit(EXIT_FAILURE);
  }

  // Result shape: [out_features]
  int* result_shape = (int*)malloc(1 * sizeof(int));
  if (result_shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(w_float); free(x_float);
    if (b_float) free(b_float);
    exit(EXIT_FAILURE);
  }
  result_shape[0] = weights->shape[0];
  size_t result_size = result_shape[0];

  float* out = (float*)calloc(result_size, sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(w_float); free(x_float);
    if (b_float) free(b_float);
    free(result_shape);
    exit(EXIT_FAILURE);
  }

  // Reshape input to 2D for matrix multiplication: [1, in_features]
  int input_shape_2d[2] = {1, input->shape[0]}, weight_shape[2] = {weights->shape[0], weights->shape[1]};
  // Perform matrix multiplication: W @ x (reshaped as column vector)
  // We need to transpose input from [1, in_features] to [in_features, 1] for W @ x
  float* x_reshaped = (float*)malloc(input->shape[0] * sizeof(float));
  if (x_reshaped == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(w_float);
    free(x_float);
    if (b_float) free(b_float);
    free(result_shape);
    free(out);
    exit(EXIT_FAILURE);
  }

  // Copy input data (already in correct format for column vector)
  for (int i = 0; i < input->shape[0]; i++) { x_reshaped[i] = x_float[i]; }
  for (int i = 0; i < weights->shape[0]; i++) { dot_tensor_ops(w_float + i * weights->shape[1], x_reshaped, out + i, weights->shape[1]); }
  free(x_reshaped);

  if (bias != NULL) { add_bias_linear_ops(out, b_float, result_shape, 1); } // Add bias if provided
  dtype_t result_dtype = promote_dtypes(weights->dtype, input->dtype);  // Determine result dtype
  if (bias != NULL) { result_dtype = promote_dtypes(result_dtype, bias->dtype); }
  Tensor* result = create_tensor(out, 1, result_shape, result_size, result_dtype);
  free(w_float); free(x_float);
  if (b_float) free(b_float);
  free(out); free(result_shape);
  return result;
}

// Linear transformation for 2D input (batch processing): Y = XW^T + b
// weights: [out_features, in_features], input: [batch_size, in_features], bias: [out_features] (optional)
// Returns: [batch_size, out_features]
Tensor* linear_2d_tensor(Tensor* weights, Tensor* input, Tensor* bias) {
  if (weights == NULL || input == NULL) {
    fprintf(stderr, "Weight and input tensors cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (weights->shape[1] != input->shape[1]) {
    fprintf(stderr, "Weight columns must match input features: %d != %d\n", weights->shape[1], input->shape[1]);
    exit(EXIT_FAILURE);
  }
  if (bias != NULL && (bias->ndim != 1 || bias->shape[0] != weights->shape[0])) {
    fprintf(stderr, "Bias must be 1D with size matching weight rows: %d\n", weights->shape[0]);
    exit(EXIT_FAILURE);
  }

  float* w_float = convert_to_float32(weights->data, weights->dtype, weights->size);
  float* x_float = convert_to_float32(input->data, input->dtype, input->size);
  float* b_float = NULL;
  if (bias != NULL) { b_float = convert_to_float32(bias->data, bias->dtype, bias->size); }

  if (w_float == NULL || x_float == NULL || (bias != NULL && b_float == NULL)) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (w_float) free(w_float);
    if (x_float) free(x_float);
    if (b_float) free(b_float);
    exit(EXIT_FAILURE);
  }

  int* result_shape = (int*)malloc(2 * sizeof(int));
  if (result_shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(w_float); free(x_float);
    if (b_float) free(b_float);
    exit(EXIT_FAILURE);
  }
  result_shape[0] = input->shape[0], result_shape[1] = weights->shape[0];
  size_t result_size = result_shape[0] * result_shape[1];

  float* out = (float*)calloc(result_size, sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(w_float); free(x_float);
    if (b_float) free(b_float);
    free(result_shape);
    exit(EXIT_FAILURE);
  }

  float* w_transposed = (float*)malloc(weights->shape[0] * weights->shape[1] * sizeof(float));
  if (w_transposed == NULL) {
    fprintf(stderr, "Memory allocation failed for weight transpose\n");
    free(w_float); free(x_float);
    if (b_float) free(b_float);
    free(result_shape); free(out);
    exit(EXIT_FAILURE);
  }

  int weight_shape[2] = {weights->shape[0], weights->shape[1]};
  transpose_2d_tensor_ops(w_float, w_transposed, weight_shape);
  
  int input_shape[2] = {input->shape[0], input->shape[1]};
  int transposed_weight_shape[2] = {weights->shape[1], weights->shape[0]};
  matmul_tensor_ops(x_float, w_transposed, out, input_shape, transposed_weight_shape);
  
  if (bias != NULL) { add_bias_linear_ops(out, b_float, result_shape, 2); }
  dtype_t result_dtype = promote_dtypes(weights->dtype, input->dtype);
  if (bias != NULL) { result_dtype = promote_dtypes(result_dtype, bias->dtype); }
  Tensor* result = create_tensor(out, 2, result_shape, result_size, result_dtype);
  free(w_float); free(x_float); free(w_transposed);
  if (b_float) free(b_float);
  free(out); free(result_shape);
  return result;
}

// General linear transformation function that handles both 1D and 2D inputs
// Automatically dispatches to appropriate specialized function
Tensor* linear_transform_tensor(Tensor* weights, Tensor* input, Tensor* bias) {
  if (input == NULL || weights == NULL) {
    fprintf(stderr, "Input and weight tensors cannot be null!\n");
    exit(EXIT_FAILURE);
  }

  if (input->ndim == 1 || weights->ndim == 1) { return linear_1d_tensor(weights, input, bias); }
  else if (input->ndim == 2 || weights->ndim == 2) { return linear_2d_tensor(weights, input, bias); }
  else {
    fprintf(stderr, "Linear transformation only supports 1D or 2D input tensors\n");
    exit(EXIT_FAILURE);
  }
}