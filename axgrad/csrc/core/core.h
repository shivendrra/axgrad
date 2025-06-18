/**
  @file core.h header file for core.cpp & tensor
  * contains core components & functions for tensor creation/deletion
  * entry point to all the tensor functions
  * includes only basic core functionalities, ops are on different file
  * compile it as:
    *- '.so': g++ -shared -fPIC -o libtensor.so core/core.cpp core/dtype.cpp tensor.cpp cpu/maths_ops.cpp cpu/helpers.cpp cpu/utils.cpp cpu/red_ops.cpp cpu/binary_ops.cpp
    *- '.dll': g++ -shared -o libtensor.dll core/core.cpp core/dtype.cpp tensor.cpp cpu/maths_ops.cpp cpu/helpers.cpp cpu/utils.cpp cpu/red_ops.cpp cpu/binary_ops.cpp
    *- '.dylib': g++ -dynamiclib -o libtensor.dylib core/core.cpp core/dtype.cpp tensor.cpp cpu/maths_ops.cpp cpu/helpers.cpp cpu/utils.cpp cpu/red_ops.cpp cpu/binary_ops.cpp
*/

#ifndef __CORE__H__
#define __CORE__H__

#include <stddef.h>
#include "dtype.h"

typedef struct Tensor {
  void *data;   // raw data pointer (can be any dtype)
  int *shape, *strides;
  int ndim, size;
  int is_view;  // flag to indicate if this is a view of another array
  dtype_t dtype; // data type of the array
} Tensor;

extern "C" {
  // tensor initialization & deletion related function
  Tensor* create_tensor(float* data, size_t ndim, int* shape, size_t size, dtype_t dtype);
  void delete_tensor(Tensor* self);
  void delete_shape(Tensor* self);
  void delete_data(Tensor* self);
  void delete_strides(Tensor* self);
  void print_tensor(Tensor* self);

  // data returning function for python endpoint
  float* out_data(Tensor* self);
  int* out_shape(Tensor* self);
  int* out_strides(Tensor* self);
  int out_size(Tensor* self);

  // contiguous tensor related ops
  Tensor* contiguous_tensor(Tensor* self);
  void make_contiguous_inplace_tensor(Tensor* self);
  int is_contiguous_tensor(Tensor* self);

  // view/reshaping
  int is_view_tensor(Tensor* self);
  Tensor* view_tensor(Tensor* self);
  Tensor* reshape_view(Tensor* self, int* new_shape, size_t new_ndim);
  Tensor* slice_view(Tensor* self, int* start, int* end, int* step);
  Tensor* copy_tensor(Tensor* self);

  // dtype casting related functions
  Tensor* cast_tensor_simple(Tensor* self, dtype_t new_dtype);
  Tensor* cast_tensor(Tensor* self, dtype_t new_dtype);
}

#endif  //!__CORE__H__