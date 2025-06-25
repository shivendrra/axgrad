#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include "ops_binary.h"

void add_ops(float* a, float* b, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) { out[i] = a[i] + b[i]; }
}

void add_scalar_ops(float* a, float b, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) { out[i] = a[i] + b; }
}

void sub_ops(float* a, float* b, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) { out[i] = a[i] - b[i]; }
}

void sub_scalar_ops(float* a, float b, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) { out[i] = a[i] - b; }
}

void mul_ops(float* a, float* b, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) { out[i] = a[i] * b[i]; }
}

void mul_scalar_ops(float* a, float b, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) { out[i] = a[i] * b; }
}

void div_ops(float* a, float* b, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    if (b[i] == 0.0f) {
      if (a[i] > 0.0f) {
        out[i] = INFINITY;
      } else if (a[i] < 0.0f) {
        out[i] = -INFINITY;
      } else {
        out[i] = NAN;  // 0/0 case
      }
    } else {
      out[i] = a[i] / b[i];
    }
  }
}

void div_scalar_ops(float* a, float b, float* out, size_t size) {
  if (b == 0.0f) {
    for (size_t i = 0; i < size; i++) {
      if (a[i] > 0.0f) {
        out[i] = INFINITY;
      } else if (a[i] < 0.0f) {
        out[i] = -INFINITY;
      } else {
        out[i] = NAN;  // 0/0 case
      }
    }
  } else {
    for (size_t i = 0; i < size; i++) {
      out[i] = a[i] / b;
    }
  }
}

void pow_tensor_ops(float* a, float exp, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) { out[i] = powf(a[i], exp); }
}

void pow_scalar_ops(float a, float* exp, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) { out[i] = powf(a, exp[i]); }
}

void add_broadcasted_tensor_ops(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape) {
  int max_ndim = a_ndim > b_ndim ? b_ndim : a_ndim;

  int* strides1 = (int*)malloc(max_ndim * sizeof(int));
  int* strides2 = (int*)malloc(max_ndim * sizeof(int));
  if (strides1 == NULL || strides2 == NULL) {
    fprintf(stderr, "Couldn't assign the strides to memory, operation failed!\n");
    exit(1);
  }
  int stride1 = 1, stride2 = 1;
  for (int i = max_ndim; i >=0 ; i--) {
    int dim1 = i<a_ndim ? a_shape[a_ndim - max_ndim + i] : 1;
    int dim2 = i<b_ndim ? b_shape[b_ndim - max_ndim + i] : 1;
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
    out[i] = a[index1] + b[index2];
  }
  free(strides1);
  free(strides2);
}

void sub_broadcasted_tensor_ops(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape) {
  int max_ndim = a_ndim > b_ndim ? b_ndim : a_ndim;

  int* strides1 = (int*)malloc(max_ndim * sizeof(int));
  int* strides2 = (int*)malloc(max_ndim * sizeof(int));
  if (strides1 == NULL || strides2 == NULL) {
    fprintf(stderr, "Couldn't assign the strides to memory, operation failed!\n");
    exit(1);
  }
  int stride1 = 1, stride2 = 1;
  for (int i = max_ndim; i >=0 ; i--) {
    int dim1 = i<a_ndim ? a_shape[a_ndim - max_ndim + i] : 1;
    int dim2 = i<b_ndim ? b_shape[b_ndim - max_ndim + i] : 1;
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
    out[i] = a[index1] - b[index2];
  }
  free(strides1);
  free(strides2);
}

void mul_broadcasted_tensor_ops(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape) {
  int max_ndim = a_ndim > b_ndim ? b_ndim : a_ndim;

  int* strides1 = (int*)malloc(max_ndim * sizeof(int));
  int* strides2 = (int*)malloc(max_ndim * sizeof(int));
  if (strides1 == NULL || strides2 == NULL) {
    fprintf(stderr, "Couldn't assign the strides to memory, operation failed!\n");
    exit(1);
  }
  int stride1 = 1, stride2 = 1;
  for (int i = max_ndim; i >=0 ; i--) {
    int dim1 = i<a_ndim ? a_shape[a_ndim - max_ndim + i] : 1;
    int dim2 = i<b_ndim ? b_shape[b_ndim - max_ndim + i] : 1;
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
    out[i] = a[index1] * b[index2];
  }
  free(strides1);
  free(strides2);
}

void div_broadcasted_tensor_ops(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape) {
  int max_ndim = a_ndim > b_ndim ? b_ndim : a_ndim;

  int* strides1 = (int*)malloc(max_ndim * sizeof(int));
  int* strides2 = (int*)malloc(max_ndim * sizeof(int));
  if (strides1 == NULL || strides2 == NULL) {
    fprintf(stderr, "Couldn't assign the strides to memory, operation failed!\n");
    exit(1);
  }
  int stride1 = 1, stride2 = 1;
  for (int i = max_ndim; i >=0 ; i--) {
    int dim1 = i<a_ndim ? a_shape[a_ndim - max_ndim + i] : 1;
    int dim2 = i<b_ndim ? b_shape[b_ndim - max_ndim + i] : 1;
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
    if (b[index2] == 0.0f) {
      if (a[index1] > 0.0f) {
        out[i] = INFINITY;
      } else if (a[index1] < 0.0f) {
        out[i] = -INFINITY;
      } else {
        out[i] = NAN;  // 0/0 case
      }
    } else {
      out[i] = a[index1] / b[index2];
    }
  }
  free(strides1);
  free(strides2);
}