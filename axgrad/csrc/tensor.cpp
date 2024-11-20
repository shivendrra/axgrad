#include "tensor.h"
#include "cpu.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

Tensor* create_tensor(float* data, int* shape, int ndim) {
  Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
  if (tensor == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  tensor->data = data;
  tensor->shape = shape;
  tensor->ndim = ndim;

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

Tensor* add_tensor(Tensor* a, Tensor* b) {

}