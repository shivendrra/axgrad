#ifndef __FUNC_BACK__H__
#define __FUNC_BACK__H__

#include "../../csrc/core/core.h"
#include "../../csrc/core/dtype.h"

extern "C" {
  Tensor* sin_backwards(Tensor* a);
  Tensor* cos_backwards(Tensor* a);
  Tensor* tan_backwards(Tensor* a);
  Tensor* sinh_backwards(Tensor* a);
  Tensor* cosh_backwards(Tensor* a);
  Tensor* tanh_backwards(Tensor* out);
  Tensor* sigmoid_backwards(Tensor* out);
  Tensor* relu_backwards(Tensor* out);
  Tensor* elu_backwards(Tensor* a, float alpha);
  Tensor* leaky_relu_backwards(Tensor* a, float eps);
  Tensor* gelu_backwards(Tensor* a);
  Tensor* swish_backwards(Tensor* a, float beta);
  Tensor* silu_backwards(Tensor* a);
  Tensor* softplus_backwards(Tensor* a);
}

#endif  //!__FUNC_BACK__H__