#ifndef __TRANSFORM__H__
#define __TRANSFORM__H__

#include "../core/core.h"

extern "C" {
  Tensor* linear_1d_tensor(Tensor* weight, Tensor* input, Tensor* bias);
  Tensor* linear_2d_tensor(Tensor* weight, Tensor* input, Tensor* bias);
  Tensor* linear_transform_tensor(Tensor* weights, Tensor* input, Tensor* bias);
}

#endif  //!__TRANSFORM__H__