#ifndef __SHAPE_OPS__H__
#define __SHAPE_OPS__H__

#include "core/core.h"
#include "core/dtype.h"

extern "C" {
  // shaping ops
  Tensor* transpose_tensor(Tensor* a);
  Tensor* equal_tensor(Tensor* a, Tensor* b);
  Tensor* reshape_tensor(Tensor* a, int* new_shape, int new_ndim);
  Tensor* squeeze_tensor(Tensor* a, int axis);
  Tensor* expand_dims_tensor(Tensor* a, int axis);
  Tensor* flatten_tensor(Tensor* a);
}

#endif  //!__SHAPE_OPS__H__