#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "ops_redux.h"

void max_tensor_ops(float* a, float* out, size_t size, int* shape, int* strides, int* res_shape, int axis, int ndim) {
  if (axis == -1) {
    // global max - find maximum of all elements
    float max_val = a[0];  // initialize with first element instead of INFINITY
    for (int i = 1; i < size; i++) max_val = fmax(max_val, a[i]);
    *out = max_val;
  } else {
    // axis-specific max
    if (axis < 0 || axis >= ndim) { printf("Invalid axis\n"); return; }
    int out_size = 1; // calculate output size (product of all dimensions except the axis dimension)
    for (int i = 0; i < ndim; i++) { if (i != axis) out_size *= shape[i]; }
    for (int i = 0; i < out_size; i++) out[i] = INFINITY; // initialize output tensor to positive infinity
    for (int i = 0; i < size; i++) {      // iterate through all elements in the input tensor
      int coords[ndim], temp_i = i;   // convert linear index to multi-dimensional coordinates
      for (int d = ndim - 1; d >= 0; d--) {
        coords[d] = temp_i % shape[d];
        temp_i /= shape[d];
      }
      int out_idx = 0, multiplier = 1, res_dim = 0;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != axis) {
          out_idx += coords[d] * multiplier;
          multiplier *= res_shape[res_dim];
        }
      }      
     out[out_idx] = fmax(out[out_idx], a[i]);
    }
  }
}

void min_tensor_ops(float* a, float* out, size_t size, int* shape, int* strides, int* res_shape, int axis, int ndim) {
  if (axis == -1) {
    // global min - find minimum of all elements
    float min_val = a[0];  // initialize with first element instead of INFINITY
    for (int i = 1; i < size; i++) min_val = fmin(min_val, a[i]);
    *out = min_val;
  } else {
    if (axis < 0 || axis >= ndim) { printf("Invalid axis\n"); return; }
    int out_size = 1;
    for (int i = 0; i < ndim; i++) { if (i != axis) out_size *= shape[i]; }
    for (int i = 0; i < out_size; i++) out[i] = INFINITY;
    for (int i = 0; i < size; i++) {
      int coords[ndim], temp_i = i;
      for (int d = ndim - 1; d >= 0; d--) {
        coords[d] = temp_i % shape[d];
        temp_i /= shape[d];
      }

      int out_idx = 0, multiplier = 1, res_dim = 0;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != axis) {
          out_idx += coords[d] * multiplier;
          multiplier *= res_shape[res_dim];
        }
      }      
     out[out_idx] = fmin(out[out_idx], a[i]);
    }
  }
}

void sum_tensor_ops(float* a, float* out, int* shape, int* strides, int size, int* res_shape, int axis, int ndim) {
  if (axis == -1) {
    // global sum - sum all elements
    float sum = 0.0;
    for (int i = 0; i < size; i++) sum += a[i];
    *out = sum;
  } else {
    // axis-specific sum
    if (axis < 0 || axis >= ndim) { printf("Invalid Axis\n"); return; }
    // calculate output size (product of all dimensions except the axis dimension)
    int out_size = 1;
    for (int i = 0; i < ndim; i++) { if (i != axis) out_size *= shape[i]; }
    for (int i = 0; i < out_size; i++) out[i] = 0.0;
    for (int i = 0; i < size; i++) {
      int coords[ndim], temp_i = i;
      for (int d = ndim - 1; d >= 0; d--) {
        coords[d] = temp_i % shape[d];
        temp_i /= shape[d];
      }
      int out_idx = 0, multiplier = 1, res_dim = 0;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != axis) {
          out_idx += coords[d] * multiplier;
          multiplier *= res_shape[res_dim];
        }
      }
      out[out_idx] += a[i];
    }
  }
}

void mean_tensor_ops(float* a, float* out, int* shape, int* strides, int size, int* res_shape, int axis, int ndim) {
  if (axis == -1) {
    float sum = 0.0;
    for (int i = 0; i < size; i++) sum += a[i];
    *out = sum / size;
  } else {
    if (axis < 0 || axis >= ndim) { printf("Invalid Axis\n"); return; }
    int out_size = 1;
    for (int i = 0; i < ndim; i++) { if (i != axis) out_size *= shape[i]; }
    for (int i = 0; i < out_size; i++) out[i] = 0.0;
    for (int i = 0; i < size; i++) {
      int coords[ndim], temp_i = i;
      for (int d = ndim - 1; d >= 0; d--) {
        coords[d] = temp_i % shape[d];
        temp_i /= shape[d];
      }
      int out_idx = 0, multiplier = 1, res_dim = 0;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != axis) {
          out_idx += coords[d] * multiplier;
          multiplier *= res_shape[res_dim];
        }
      }
      out[out_idx] += a[i];
    }
    int axis_size = shape[axis];
    for (int i = 0; i < out_size; i++) out[i] /= axis_size;
  }
}

void var_tensor_ops(float* a, float* out, size_t size, int* shape, int* strides, int* res_shape, int axis, int ndim, int ddof) {
  if (axis == -1) {
    // global variance - calculate variance of all elements
    float mean = 0.0;   // first pass: calculate mean
    for (int i = 0; i < size; i++) mean += a[i];
    mean /= size;
    float variance = 0.0;     // second pass: calculate variance
    for (int i = 0; i < size; i++) {
      float diff = a[i] - mean;
      variance += diff * diff;
    }

    // divide by (N - ddof) for sample variance, or N for population variance
    int denominator = size - ddof;
    if (denominator <= 0) { printf("Warning: ddof >= sample size, setting variance to 0\n"); *out = 0.0; }
    else { *out = variance / denominator; }
  } else {
    if (axis < 0 || axis >= ndim) { printf("Invalid axis\n"); return; }
    int out_size = 1;
    for (int i = 0; i < ndim; i++) { if (i != axis) out_size *= shape[i]; }
    int axis_size = shape[axis];
    float* means = (float*)calloc(out_size, sizeof(float));
    for (int i = 0; i < out_size; i++) out[i] = 0.0;
    for (int i = 0; i < size; i++) {
      int coords[ndim], temp_i = i;
      for (int d = ndim - 1; d >= 0; d--) {
        coords[d] = temp_i % shape[d];
        temp_i /= shape[d];
      }      
      int out_idx = 0, multiplier = 1, res_dim = 0;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != axis) {
          out_idx += coords[d] * multiplier;
          multiplier *= res_shape[res_dim];
        }
      }
      means[out_idx] += a[i];       // accumulate sum for mean calculation
    }
    for (int i = 0; i < out_size; i++) means[i] /= axis_size;
    // second pass: calculate variance for each output position
    for (int i = 0; i < size; i++) {
      // convert linear index to multi-dimensional coordinates
      int coords[ndim], temp_i = i;
      for (int d = ndim - 1; d >= 0; d--) {
        coords[d] = temp_i % shape[d];
        temp_i /= shape[d];
      }

      int out_idx = 0, multiplier = 1, res_dim = 0;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != axis) {
          out_idx += coords[d] * multiplier;
          multiplier *= res_shape[res_dim];
        }
      }
      float diff = a[i] - means[out_idx]; // accumulate squared differences
      out[out_idx] += diff * diff;
    }

    // divide by (axis_size - ddof) to get final variance
    int denominator = axis_size - ddof;
    if (denominator <= 0) {
      printf("Warning: ddof >= sample size, setting variance to 0\n");
      for (int i = 0; i < out_size; i++) out[i] = 0.0;
    } else { for (int i = 0; i < out_size; i++) out[i] /= denominator; }
    free(means);
  }
}

void std_tensor_ops(float* a, float* out, size_t size, int* shape, int* strides, int* res_shape, int axis, int ndim, int ddof) {
  if (axis == -1) {
    // global standard deviation - calculate std of all elements
    float mean = 0.0; // first pass: calculate mean
    for (int i = 0; i < size; i++) mean += a[i];
    mean /= size;
    float variance = 0.0; // second pass: calculate variance
    for (int i = 0; i < size; i++) {
      float diff = a[i] - mean;
      variance += diff * diff;
    }

    // divide by (N - ddof) for sample variance, or N for population variance
    int denominator = size - ddof;
    if (denominator <= 0) {
      printf("Warning: ddof >= sample size, setting std to 0\n");
      *out = 0.0;
    } else { *out = sqrtf(variance / denominator); }
  } else {
    // axis-specific standard deviation
    if (axis < 0 || axis >= ndim) { printf("Invalid axis\n"); return; }
    // calculate output size (product of all dimensions except the axis dimension)
    int out_size = 1;
    for (int i = 0; i < ndim; i++) {
      if (i != axis) {
        out_size *= shape[i];
      }
    }
    int axis_size = shape[axis];
    // initialize output tensors
    float* means = (float*)calloc(out_size, sizeof(float));
    if (means == NULL) {
      printf("Memory allocation failed for means\n");
      return;
    }

    for (int i = 0; i < out_size; i++) {
      out[i] = 0.0;
    }

    // first pass: calculate means for each output position
    for (int i = 0; i < size; i++) {
      // convert linear index to multi-dimensional coordinates
      int coords[ndim];
      int temp_i = i;
      for (int d = ndim - 1; d >= 0; d--) {
        coords[d] = temp_i % shape[d];
        temp_i /= shape[d];
      }
      // calculate output index by removing the axis dimension
      int out_idx = 0, multiplier = 1, res_dim = 0;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != axis) {
          out_idx += coords[d] * multiplier;
          multiplier *= res_shape[res_dim];
        }
      }
      // accumulate sum for mean calculation
      means[out_idx] += a[i];
    }

    // divide by axis size to get means
    for (int i = 0; i < out_size; i++) means[i] /= axis_size;

    // second pass: calculate variance for each output position
    for (int i = 0; i < size; i++) {
      int coords[ndim];
      int temp_i = i;
      for (int d = ndim - 1; d >= 0; d--) {
        coords[d] = temp_i % shape[d];
        temp_i /= shape[d];
      }
      int out_idx = 0, multiplier = 1, res_dim = 0;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != axis) {
          out_idx += coords[d] * multiplier;
          multiplier *= res_shape[res_dim];
        }
      }
      // accumulate squared differences
      float diff = a[i] - means[out_idx];
      out[out_idx] += diff * diff;
    }
    // divide by (axis_size - ddof) and take square root to get final standard deviation
    int denominator = axis_size - ddof;
    if (denominator <= 0) {
      printf("Warning: ddof >= sample size, setting std to 0\n");
      for (int i = 0; i < out_size; i++) out[i] = 0.0;
    } else {
      for (int i = 0; i < out_size; i++) out[i] = sqrtf(out[i] / denominator);
    }

    free(means);
  }
}