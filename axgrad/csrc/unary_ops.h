#ifndef __UNARY_OPS__H__
#define __UNARY_OPS__H__

#include "core/core.h"
#include "core/dtype.h"

extern "C" {
  // unary ops
  Tensor* sin_tensor(Tensor* a);
  Tensor* sinh_tensor(Tensor* a);
  Tensor* cos_tensor(Tensor* a);
  Tensor* cosh_tensor(Tensor* a);
  Tensor* tan_tensor(Tensor* a);
  Tensor* tanh_tensor(Tensor* a);
  Tensor* log_tensor(Tensor* a);
  Tensor* exp_tensor(Tensor* a);
  Tensor* abs_tensor(Tensor* a);
  Tensor* neg_tensor(Tensor* a);
  Tensor* sqrt_tensor(Tensor* a);
}

#endif  //!__UNARY_OPS__H__