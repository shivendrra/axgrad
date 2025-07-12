#ifndef __FUNCTIONAL__H__
#define __FUNCTIONAL__H__

#include "core/core.h"
#include "core/dtype.h"

extern "C" {
  Tensor* sigmoid_tensor(Tensor* a);
  Tensor* relu_tensor(Tensor* a);
  Tensor* gelu_tensor(Tensor* a);
  Tensor* leaky_relu_tensor(Tensor* a, float eps);
  Tensor* silu_tensor(Tensor* a);
  Tensor* elu_tensor(Tensor* a, float alpha);
  Tensor* swish_tensor(Tensor* a, float beta);
  Tensor* softplus_tensor(Tensor* a);
}

#endif  //!__FUNCTIONAL__H__