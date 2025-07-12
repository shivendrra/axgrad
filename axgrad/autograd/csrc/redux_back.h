#ifndef __REDUX_BACK__H__
#define __REDUX_BACK__H__

#include "../../csrc/core/core.h"
#include "../../csrc/core/dtype.h"

extern "C" {
  Tensor* sum_backwards(Tensor* grad_output, int* original_shape, int original_ndim, size_t original_size, int axis);
  Tensor* mean_backwards(Tensor* grad_output, int* original_shape, int original_ndim, size_t original_size, int axis);
  Tensor* var_backwards(Tensor* a, Tensor* grad_output, int* original_shape, int original_ndim, size_t original_size, int axis, int ddof);
  Tensor* std_backwards(Tensor* a, Tensor* grad_output, int* original_shape, int original_ndim, size_t original_size, int axis, int ddof);
}

#endif  //!__REDUX_BACK__H__