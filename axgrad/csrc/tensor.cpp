#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include "tensor.h"
#include "cuda.h"
#include "cpu.h"

extern "C" {
  Tensor* create_tensor(float* data, int* shape, int ndim) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (tensor == NULL){
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    tensor->data = data;
    tensor->shape = shape;
    tensor->ndim = ndim;

    tensor->size = 1;
    for(int i = 0; i < ndim; i++){
      tensor->size *= shape[i];
    }

    tensor->strides = (int*)malloc(ndim*sizeof(int));
    if (tensor->strides == NULL){
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    int stride = 1;
    for (int i = ndim-1; i>=0; i--){
      tensor->strides[i] = stride;
      stride *= shape[i];
    }
    return tensor;
  }

  float get_item(Tensor* tensor, int* indices) {
    int index = 0;
    for (int i = 0; i < tensor->ndim; i++) {
      index += indices[i] * tensor->strides[i];
    }

    float result;
    result = tensor->data[index];

    return result;
  }
}