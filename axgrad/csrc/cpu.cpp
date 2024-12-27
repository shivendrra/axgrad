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

void add_broadcasted_tensor_cpu(Tensor* a, Tensor* b, float* out, int* broadcasted_shape, int broadcasted_size) {
  int max_ndim = a->ndim > b->ndim ? b->ndim : a->ndim;

  int* strides1 = (int*)malloc(max_ndim * sizeof(int));
  int* strides2 = (int*)malloc(max_ndim * sizeof(int));
  if (strides1 == NULL || strides2 == NULL) {
    fprintf(stderr, "Couldn't assign the strides to memory, operation failed!\n");
    exit(1);
  }
  int stride1 = 1, stride2 = 1;
  for (int i = max_ndim; i >=0 ; i--) {
    int dim1 = i<a->ndim ? a->shape[a->ndim - max_ndim + i] : 1;
    int dim2 = i<b->ndim ? b->shape[b->ndim - max_ndim + i] : 1;
    strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
    strides2[i] = dim1 == broadcasted_shape[i] ? stride2 : 0;
    stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
    stride2 *= (dim1 == broadcasted_shape[i]) ? dim2 : 1;
  }

  for (int i = 0; i < broadcasted_size; i++) {
    int index1 = 0, index2 = 0;
    int linear_index = i;
    for (int j = max_ndim - 1; j >= 0; j--) {
      int pos = linear_index % broadcasted_shape[j];
      linear_index /= broadcasted_shape[j];
      if (strides1[j] != 0) index1 += pos * strides1[j];
      if (strides2[j] != 0) index2 += pos * strides2[j];
    }
    out[i] = a->data[i] + b->data[i];
  }
  free(strides1);
  free(strides2);
}

void sub_broadcasted_tensor_cpu(Tensor* a, Tensor* b, float* out, int* broadcasted_shape, int broadcasted_size) {
  int max_ndim = a->ndim > b->ndim ? b->ndim : a->ndim;

  int* strides1 = (int*)malloc(max_ndim * sizeof(int));
  int* strides2 = (int*)malloc(max_ndim * sizeof(int));
  if (strides1 == NULL || strides2 == NULL) {
    fprintf(stderr, "Couldn't assign the strides to memory, operation failed!\n");
    exit(1);
  }
  int stride1 = 1, stride2 = 1;
  for (int i = max_ndim; i >=0 ; i--) {
    int dim1 = i<a->ndim ? a->shape[a->ndim - max_ndim + i] : 1;
    int dim2 = i<b->ndim ? b->shape[b->ndim - max_ndim + i] : 1;
    strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
    strides2[i] = dim1 == broadcasted_shape[i] ? stride2 : 0;
    stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
    stride2 *= (dim1 == broadcasted_shape[i]) ? dim2 : 1;
  }

  for (int i = 0; i < broadcasted_size; i++) {
    int index1 = 0, index2 = 0;
    int linear_index = i;
    for (int j = max_ndim - 1; j >= 0; j--) {
      int pos = linear_index % broadcasted_shape[j];
      linear_index /= broadcasted_shape[j];
      if (strides1[j] != 0) index1 += pos * strides1[j];
      if (strides2[j] != 0) index2 += pos * strides2[j];
    }
    out[i] = a->data[i] - b->data[i];
  }
  free(strides1);
  free(strides2);
}

void mul_broadcasted_tensor_cpu(Tensor* a, Tensor* b, float* out, int* broadcasted_shape, int broadcasted_size) {
  int max_ndim = a->ndim > b->ndim ? b->ndim : a->ndim;

  int* strides1 = (int*)malloc(max_ndim * sizeof(int));
  int* strides2 = (int*)malloc(max_ndim * sizeof(int));
  if (strides1 == NULL || strides2 == NULL) {
    fprintf(stderr, "Couldn't assign the strides to memory, operation failed!\n");
    exit(1);
  }
  int stride1 = 1, stride2 = 1;
  for (int i = max_ndim; i >=0 ; i--) {
    int dim1 = i<a->ndim ? a->shape[a->ndim - max_ndim + i] : 1;
    int dim2 = i<b->ndim ? b->shape[b->ndim - max_ndim + i] : 1;
    strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
    strides2[i] = dim1 == broadcasted_shape[i] ? stride2 : 0;
    stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
    stride2 *= (dim1 == broadcasted_shape[i]) ? dim2 : 1;
  }

  for (int i = 0; i < broadcasted_size; i++) {
    int index1 = 0, index2 = 0;
    int linear_index = i;
    for (int j = max_ndim - 1; j >= 0; j--) {
      int pos = linear_index % broadcasted_shape[j];
      linear_index /= broadcasted_shape[j];
      if (strides1[j] != 0) index1 += pos * strides1[j];
      if (strides2[j] != 0) index2 += pos * strides2[j];
    }
    out[i] = a->data[i] * b->data[i];
  }
  free(strides1);
  free(strides2);
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

void matmul_tensor_cpu(Tensor* a, Tensor* b, float* out) {
  for (int i = 0; i < a->shape[0]; i++) {
    for (int j = 0; j < b->shape[1]; j++) {
      float sum = 0.0;
      for (int k = 0; k < a->shape[1]; k++) {
        sum += a->data[i * a->shape[1] + k] * b->data[k * b->shape[1] + j];
      }
      out[i * b->shape[1] + j] = sum;
    }
  }
}

void broadcasted_batched_matmul_tensor_cpu(Tensor* a, Tensor* b, float* out) {
  int out_stride = a->shape[0] * b->shape[2];
  for (int batch = 0; batch < b->shape[0]; batch++) {
    for (int i = 0; i < a->shape[0]; i++) {
      for (int j = 0; j < b->shape[2]; j++) {
        float sum = 0.0;
        for (int k = 0; k < a->shape[1]; k++) {
          sum += a->data[i * a->shape[1] + k] * b->data[batch*b->strides[0] + (k * b->shape[2] + j)];
        }
        out[(batch * out_stride) + (i * b->shape[2] + j)] = sum;
      }
    }
  }
}

void batched_matmul_tensor_cpu(Tensor* a, Tensor* b, float* out) {
  int out_stride = a->shape[1] * b->shape[2];
  for (int batch = 0; batch < b->shape[0]; batch++) {    
    for (int i = 0; i < a->shape[1]; i++) {
      for (int j = 0; j < b->shape[2]; j++) {
        float sum = 0.0;
        for (int k = 0; k < a->shape[2]; k++) {
          sum += a->data[(batch * a->strides[0]) + i * a->shape[2] + k] * b->data[batch*b->strides[0] + (k * b->shape[2] + j)];
        }
        out[(batch * out_stride) + (i * b->shape[2] + j)] = sum;
      }
    }
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

void ones_like_tensor_cpu(Tensor* a, float* out) {
  for (int i = 0; i < a->size; i++) {
    out[i] = 1.0;
  }
}

void zeros_like_tensor_cpu(Tensor* a, float* out) {
  for (int i = 0; i < a->size; i++) {
    out[i] = 0.0;
  }
}

void transpose_1d_tensor_cpu(Tensor* a, float* out) {
  for (int i = 0; i < a->shape[0]; i++) {
    out[i] = a->data[i];
  }
}

void transpose_2d_tensor_cpu(Tensor* a, float* out) {
  int rows = a->shape[0], cols = a->shape[1];
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      out[j * rows + i] = a->data[i * cols + j];
    }
  }
}

void transpose_3d_tensor_cpu(Tensor* a, float* out) {
  int batch = a->shape[0], rows = a->shape[0], cols = a->shape[1];
  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < rows; j++) {
      for (int k = 0; k < cols; k++) {
        out[k * rows * batch + j * batch + i] = a->data[i * rows * cols + j * cols + k];
      }
    }
  }
}

void reassign_tensor_cpu(Tensor* a, float* out) {
  for (int i = 0; i < a->size; i++) {
    out[i] = a->data[i];
  }
}

void make_contagious_tensor_cpu(Tensor* a, float* out, int* new_strides) {
  for (int i = 0; i < a->size; i++) {
    int index = 0, offset = i;
    for (int j = 0; j < a->ndim; j++) {
      index += (offset / new_strides[j]) * a->strides[j];
      offset %= new_strides[j];
    }
    out[i] = a->data[index];
  }
  free(a->data);
  free(a->strides);
  a->data = out, a->strides = new_strides;
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

void equal_broadcasted_tensor_cpu(Tensor* a, Tensor* b, float* out, int* broadcasted_shape, int broadcasted_size) {
  int max_ndim = a->ndim > b->ndim ? b->ndim : a->ndim;

  int* strides1 = (int*)malloc(max_ndim * sizeof(int));
  int* strides2 = (int*)malloc(max_ndim * sizeof(int));
  if (strides1 == NULL || strides2 == NULL) {
    fprintf(stderr, "Couldn't assign the strides to memory, operation failed!\n");
    exit(1);
  }
  int stride1 = 1, stride2 = 1;
  for (int i = max_ndim; i >=0 ; i--) {
    int dim1 = i<a->ndim ? a->shape[a->ndim - max_ndim + i] : 1;
    int dim2 = i<b->ndim ? b->shape[b->ndim - max_ndim + i] : 1;
    strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
    strides2[i] = dim1 == broadcasted_shape[i] ? stride2 : 0;
    stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
    stride2 *= (dim1 == broadcasted_shape[i]) ? dim2 : 1;
  }

  for (int i = 0; i < broadcasted_size; i++) {
    int index1 = 0, index2 = 0;
    int linear_index = i;
    for (int j = max_ndim - 1; j >= 0; j--) {
      int pos = linear_index % broadcasted_shape[j];
      linear_index /= broadcasted_shape[j];
      if (strides1[j] != 0) index1 += pos * strides1[j];
      if (strides2[j] != 0) index2 += pos * strides2[j];
    }
    out[i] = (a->data[index1] == b->data[index2]) ? 1.0f : 0.0f;
  }
  free(strides1);
  free(strides2);
}

void sin_tensor_cpu(Tensor* a, float* out) {
  for (int i = 0; i < a->size; i++) {
    out[i] = sinf(a->data[i]);
  }
}

void cos_tensor_cpu(Tensor* a, float* out) {
  for (int i = 0; i < a->size; i++) {
    out[i] = cosf(a->data[i]);
  }
}

void sigmoid_tensor_cpu(Tensor* a, float* out) {
  for (int i = 0; i < a->size; i++) {
    if (a->data[i] >= 0) {
      out[i] = 1 / (1 + expf(-a->data[i]));
    } else {
      out[i] = expf(a->data[i]) / (1 + expf(a->data[i]));
    }
  }
}

void tanh_tensor_cpu(Tensor* a, float* out) {
  for (int i = 0; i < a->size; i++) {
    out[i] = tanh(a->data[i]);
  }
}

void relu_tensor_cpu(Tensor* a, float* out) {
  for (int i = 0; i < a->size; i++) {
    out[i] = fmax(a->data[i], 0.0);
  }
}