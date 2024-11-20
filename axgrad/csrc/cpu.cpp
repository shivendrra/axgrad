#include "tensor.h"
#include "cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void add_tensor_cpu(Tensor* a, Tensor* b, float* out) {
  for (int i = 0; i < a->size; i++) {
    out[i] = a->data[i] + b->data[i];
  }
}

void sub_tensor_cpu(Tensor* a, Tensor* b, float* out) {
  for (int i = 0; i < a->size; i++) {
    out[i] = a->data[i] - b->data[i];
  }
}

void mul_tensor_cpu(Tensor* a, Tensor* b, float* out) {
  for (int i = 0; i < a->size; i++) {
    out[i] = a->data[i] * b->data[i];
  }
}

void div_tensor_cpu(Tensor* a, Tensor* b, float* out) {
  for (int i = 0; i < a->size; i++) {
    out[i] = a->data[i] / b->data[i];
  }
}

void scalar_mul_tensor_cpu(Tensor* a, float b, float* out) {
  for (int i = 0; i < a->size; i++) {
    out[i] = a->data[i] * b;
  }
}

void scalar_div_tensor_cpu(Tensor* a, float b, float* out) {
  for (int i = 0; i < a->size; i++) {
    out[i] = b / a->data[i];
  }
}

void tensor_div_scalar_cpu(Tensor* a, float b, float* out) {
  for (int i = 0; i < a->size; i++) {
    out[i] = a->data[i] / b;
  }
}

void tensor_pow_scalar_cpu(Tensor* a, float b, float* out) {
  for (int i = 0; i < a->size; i++) {
    out[i] = powf(a->data[i], b);
  }
}

void scalar_pow_tensor_cpu(float base, Tensor* b, float* out) {
  for (int i = 0; i < b->size; i++) {
    out[i] = powf(base, b->data[i]);
  }
}

void log_tensor_cpu(Tensor* a, float* out) {
  for (int i = 0; i < a->size; i++) {
    out[i] = logf(a->data[i]);
  }
}

void sum_tensor_cpu(Tensor* a, float* out, int size, int* res_shape, int axis) {
  if (axis == -1) {
    float sum = 0.0;
    for (int i = 0; i < a->size; i++) {
      sum += a->data[i];
    }
    *out = sum;
  } else {
    if (axis < 0 || axis >= a->ndim) {
      printf("Invalid Axis");
      return;
    }
    int axis_strides = a->strides[axis];
    for (int i = 0; a->shape[axis]; i++) {
      for (int j = 0; j < size; j++) {
        int index = 0;
        int remainder = j;
        for (int k = a->ndim - 2; k >= 0; k--) {
          index += (remainder % res_shape[k]) * a->shape[k < axis ? k : k + 1];
          remainder /= res_shape[k];
        }
        out[j] = a->data[index + i * axis_strides];
      }
    }
  }
}

void max_tensor_cpu(Tensor* a, float* out, int size, int* res_shape, int axis) {
  if (axis == -1) {
    float max_val = INFINITY;
    for (int i = 0; i < a->size; i++) {
      max_val = fmax(max_val, a->data[i]);
    }
    *out = max_val;
  } else {
    for (int i = 0; i < size; i++) {
      out[i] = INFINITY;
    }
    if (axis < 0 || axis >= a->ndim) {
      printf("Invalid axis");
      return;
    }
    int axis_strides = a->strides[axis];
    for (int i = 0; a->shape[axis]; i++) {
      for (int j = 0; j < size; j++) {
        int index = 0;
        int remainder = j;
        for (int k = a->ndim - 2; k >= 0; k--) {
          index += (remainder % res_shape[k]) * a->shape[k < axis ? k : k + 1];
          remainder /= res_shape[k];
        }
        out[j] = fmax(out[j], a->data[index + i * axis_strides]);
      }
    }
  }
}

void min_tensor_cpu(Tensor* a, float* out, int size, int* res_shape, int axis) {
  if (axis == -1) {
    float min_val = INFINITY;
    for (int i = 0; i < a->size; i++) {
      min_val = fmin(min_val, a->data[i]);
    }
    *out = min_val;
  } else {
    for (int i = 0; i < size; i++) {
      out[i] = INFINITY;
    }
    if (axis < 0 || axis >= a->ndim) {
      printf("Invalid axis");
      return;
    }
    int axis_strides = a->strides[axis];
    for (int i = 0; a->shape[axis]; i++) {
      for (int j = 0; j < size; j++) {
        int index = 0;
        int remainder = j;
        for (int k = a->ndim - 2; k >= 0; k--) {
          index += (remainder % res_shape[k]) * a->shape[k < axis ? k : k + 1];
          remainder /= res_shape[k];
        }
        out[j] = fmin(out[j], a->data[index + i * axis_strides]);
      }
    }
  }
}

void equal_tensor_cpu(Tensor* a, Tensor* b, float* out) {
  for (int i = 0; i < a->size; i++) {
    out[i] = (a->data[i] == b->data[i]) ? 1.0f : 0.0f;
  }
}