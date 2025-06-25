#ifndef __BINARY_OPS__H__
#define __BINARY_OPS__H__

#include "core/core.h"
#include "core/dtype.h"

extern "C" {
  // binary ops
  Tensor* add_tensor(Tensor* a, Tensor* b);
  Tensor* add_scalar_tensor(Tensor* a, float b);
  Tensor* sub_tensor(Tensor* a, Tensor* b);
  Tensor* sub_scalar_tensor(Tensor* a, float b);
  Tensor* mul_tensor(Tensor* a, Tensor* b);
  Tensor* mul_scalar_tensor(Tensor* a, float b);
  Tensor* div_tensor(Tensor* a, Tensor* b);
  Tensor* div_scalar_tensor(Tensor* a, float b);
  Tensor* pow_tensor(Tensor* a, float exp);
  Tensor* pow_scalar(float a, Tensor* exp);

  Tensor* add_broadcasted_tensor(Tensor* a, Tensor* b);
  Tensor* sub_broadcasted_tensor(Tensor* a, Tensor* b);
  Tensor* mul_broadcasted_tensor(Tensor* a, Tensor* b);
  Tensor* div_broadcasted_tensor(Tensor* a, Tensor* b);
}

#endif  //!__BINARY_OPS__H__