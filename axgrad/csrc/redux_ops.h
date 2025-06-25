#ifndef __REDUX_OPS__H__
#define __REDUX_OPS__H__

#include "core/core.h"
#include "core/dtype.h"

extern "C" {
  // reduction ops
  Tensor* sum_tensor(Tensor* a, int axis, bool keepdims);
  Tensor* mean_tensor(Tensor* a, int axis, bool keepdims);
  Tensor* max_tensor(Tensor* a, int axis, bool keepdims);
  Tensor* min_tensor(Tensor* a, int axis, bool keepdims);
  Tensor* var_tensor(Tensor* a, int axis, int ddof);
  Tensor* std_tensor(Tensor* a, int axis, int ddof);
}

#endif  //!__REDUX_OPS__H__