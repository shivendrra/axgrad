#include "tensor.h"
#include "dtype.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

Tensor* create_tensor(void* data, int* shape, int ndim, DType dtype) {
  Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
  if (tensor == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  tensor->data = data;
  tensor->shape = shape;
  tensor->ndim = ndim;
  tensor->dtype = dtype;

  tensor->size = 1;
  for (int i = 0; i < ndim; i++) {
    tensor->size *= shape[i];
  }

  tensor->strides = (int*)malloc(ndim * sizeof(int));
  if (tensor->strides == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  int stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    tensor->strides[i] = stride;
    stride *= shape[i];
  }

  return tensor;
}

void change_tensor_dtype(Tensor* tensor, DType new_dtype) {
  if (!tensor || !tensor->data) {
    fprintf(stderr, "Invalid tensor\n");
    return;
  }

  void* new_data = malloc(dtype_size(new_dtype) * tensor->size);
  if (!new_data) {
    fprintf(stderr, "Memory allocation failed\n");
    return;
  }

  for (int i = 0; i < tensor->size; i++) {
    double value = get_data_as_double((char*)tensor->data + i * dtype_size(tensor->dtype), tensor->dtype);
    set_data_from_double((char*)new_data + i * dtype_size(new_dtype), new_dtype, value);
  }

  free(tensor->data);
  tensor->data = new_data;
  tensor->dtype = new_dtype;
}

void delete_tensor(Tensor* tensor) {
  if (tensor != NULL) {
    free(tensor);
    tensor = NULL;
  }
}

void delete_shape(Tensor* tensor) {
  if (tensor->shape != NULL) {
    free(tensor->shape);
    tensor->shape = NULL;
  }
}

void delete_data(Tensor* tensor) {
  if (tensor->data != NULL) {
    free(tensor->data);
    tensor->data = NULL;
  }
}

void delete_strides(Tensor* tensor) {
  if (tensor->strides != NULL) {
    free(tensor->strides);
    tensor->strides = NULL;
  }
}