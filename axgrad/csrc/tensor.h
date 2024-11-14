#ifndef TENSOR_H
#define TENSOR_H

#include "dtype.h"

typedef struct Tensor {
  void* data;
  int* strides;
  int* shape;
  int ndim;
  int size;
  DType dtype;
} Tensor;

extern "C" {
  Tensor* create_tensor(void* data, int* shape, int ndim, DType dtype);
  void delete_tensor(Tensor* tensor);
  void delete_strides(Tensor* tensor);
  void delete_shape(Tensor* tensor);
  void delete_data(Tensor* tensor);
  void change_tensor_dtype(Tensor* tensor, DType new_dtype);
}

#endif