#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include "redux.h"

void sum_backwards_ops(float* grad, float* out, int* original_shape, int original_ndim, size_t original_size, int axis) {
  if (axis == -1) {
    float grad_val = grad[0];
    for (size_t i = 0; i < original_size; i++) { out[i] = grad_val; }
  } else {
    for (size_t i = 0; i < original_size; i++) {
      int coords[original_ndim];
      size_t temp_i = i;
      for (int d = original_ndim - 1; d >= 0; d--) {
        coords[d] = temp_i % original_shape[d];
        temp_i /= original_shape[d];
      }

      int grad_idx = 0, multiplier = 1;
      for (int d = original_ndim - 1; d >= 0; d--) {
        if (d != axis) {
          grad_idx += coords[d] * multiplier;
          multiplier *= original_shape[d];
        }
      }
      out[i] = grad[grad_idx];
    }
  }
}

void mean_backwards_ops(float* grad, float* out, int* original_shape, int original_ndim, size_t original_size, int axis) {
  if (axis == -1) {
    float grad_val = grad[0] / original_size;
    for (size_t i = 0; i < original_size; i++) { out[i] = grad_val; }
  } else {
    int axis_size = original_shape[axis];
    for (size_t i = 0; i < original_size; i++) {
      int coords[original_ndim];
      size_t temp_i = i;
      for (int d = original_ndim - 1; d >= 0; d--) {
        coords[d] = temp_i % original_shape[d];
        temp_i /= original_shape[d];
      }

      int grad_idx = 0, multiplier = 1;
      for (int d = original_ndim - 1; d >= 0; d--) {
        if (d != axis) {
          grad_idx += coords[d] * multiplier;
          multiplier *= original_shape[d];
        }
      }
      out[i] = grad[grad_idx] / axis_size;
    }
  }
}

void var_backwards_ops(float* a, float* grad, float* out, int* original_shape, int original_ndim, size_t original_size, int axis, int ddof) {
  if (axis == -1) {
    float mean = 0.0;
    for (size_t i = 0; i < original_size; i++) { mean += a[i]; }
    mean /= original_size;
    float grad_val = grad[0] * 2.0f / (original_size - ddof);
    for (size_t i = 0; i < original_size; i++) { out[i] = grad_val * (a[i] - mean); }
  } else {
    int axis_size = original_shape[axis];
    size_t reduced_size = original_size / axis_size;
    float* means = (float*)calloc(reduced_size, sizeof(float));

    for (size_t i = 0; i < original_size; i++) {
      int coords[original_ndim];
      size_t temp_i = i;
      for (int d = original_ndim - 1; d >= 0; d--) {
        coords[d] = temp_i % original_shape[d];
        temp_i /= original_shape[d];
      }

      int mean_idx = 0, multiplier = 1;
      for (int d = original_ndim - 1; d >= 0; d--) {
        if (d != axis) {
          mean_idx += coords[d] * multiplier;
          multiplier *= original_shape[d];
        }
      }
      means[mean_idx] += a[i];
    }

    for (size_t i = 0; i < reduced_size; i++) { means[i] /= axis_size; }
    for (size_t i = 0; i < original_size; i++) {
      int coords[original_ndim];
      size_t temp_i = i;
      for (int d = original_ndim - 1; d >= 0; d--) {
        coords[d] = temp_i % original_shape[d];
        temp_i /= original_shape[d];
      }

      int grad_idx = 0, multiplier = 1;
      for (int d = original_ndim - 1; d >= 0; d--) {
        if (d != axis) {
          grad_idx += coords[d] * multiplier;
          multiplier *= original_shape[d];
        }
      }
      out[i] = grad[grad_idx] * 2.0f * (a[i] - means[grad_idx]) / (axis_size - ddof);
    }
    free(means);
  }
}

void std_backwards_ops(float* a, float* grad, float* out, int* original_shape, int original_ndim, size_t original_size, int axis, int ddof) {
  if (axis == -1) {
    float mean = 0.0;
    for (size_t i = 0; i < original_size; i++) { mean += a[i]; }
    mean /= original_size;
    float variance = 0.0;
    for (size_t i = 0; i < original_size; i++) {
      float diff = a[i] - mean;
      variance += diff * diff;
    }
    variance /= (original_size - ddof);
    float std_val = sqrtf(variance);
    float grad_val = grad[0] / (std_val * (original_size - ddof));
    for (size_t i = 0; i < original_size; i++) { out[i] = grad_val * (a[i] - mean); }
  } else {
    int axis_size = original_shape[axis];
    size_t reduced_size = original_size / axis_size;
    float *means = (float*)calloc(reduced_size, sizeof(float)), *stds = (float*)calloc(reduced_size, sizeof(float));
    for (size_t i = 0; i < original_size; i++) {
      int coords[original_ndim];
      size_t temp_i = i;
      for (int d = original_ndim - 1; d >= 0; d--) {
        coords[d] = temp_i % original_shape[d];
        temp_i /= original_shape[d];
      }

      int mean_idx = 0, multiplier = 1;
      for (int d = original_ndim - 1; d >= 0; d--) {
        if (d != axis) {
          mean_idx += coords[d] * multiplier;
          multiplier *= original_shape[d];
        }
      }
      means[mean_idx] += a[i];
    }

    for (size_t i = 0; i < reduced_size; i++) { means[i] /= axis_size; }
    for (size_t i = 0; i < original_size; i++) {
      int coords[original_ndim];
      size_t temp_i = i;
      for (int d = original_ndim - 1; d >= 0; d--) {
        coords[d] = temp_i % original_shape[d];
        temp_i /= original_shape[d];
      }

      int var_idx = 0, multiplier = 1;
      for (int d = original_ndim - 1; d >= 0; d--) {
        if (d != axis) {
          var_idx += coords[d] * multiplier;
          multiplier *= original_shape[d];
        }
      }
      float diff = a[i] - means[var_idx];
      stds[var_idx] += diff * diff;
    }

    for (size_t i = 0; i < reduced_size; i++) { stds[i] = sqrtf(stds[i] / (axis_size - ddof)); }
    for (size_t i = 0; i < original_size; i++) {
      int coords[original_ndim];
      size_t temp_i = i;
      for (int d = original_ndim - 1; d >= 0; d--) {
        coords[d] = temp_i % original_shape[d];
        temp_i /= original_shape[d];
      }

      int grad_idx = 0, multiplier = 1;
      for (int d = original_ndim - 1; d >= 0; d--) {
        if (d != axis) {
          grad_idx += coords[d] * multiplier;
          multiplier *= original_shape[d];
        }
      }
      out[i] = grad[grad_idx] * (a[i] - means[grad_idx]) / (stds[grad_idx] * (axis_size - ddof));
    }
    free(means);
    free(stds);
  }
}