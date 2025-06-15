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
  void delete_array(Tensor* self);
  void delete_shape(Tensor* self);
  void delete_data(Tensor* self);
  void delete_strides(Tensor* self);
  void print_array(Tensor* self);
}

#endif  //!__CORE__H__