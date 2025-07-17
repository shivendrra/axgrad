#ifndef __BASIC_BACK__H__
#define __BASIC_BACK__H__

#include "../../csrc/core/core.h"
#include "../../csrc/core/dtype.h"

extern "C" {
  Tensor* log_backwards(Tensor* a, Tensor* grad);
  Tensor* sqrt_backwards(Tensor* a, Tensor* grad);
  Tensor* exp_backwards(Tensor* a, Tensor* grad);
  Tensor* abs_backwards(Tensor* a, Tensor* grad);
  Tensor* neg_backwards(Tensor* a, Tensor* grad);
  Tensor* add_backwards(Tensor* grad);
  Tensor* sub_backwards(Tensor* grad, bool is_first);
  Tensor* mul_backwards(Tensor* other, Tensor* grad);
  Tensor* div_backwards(Tensor* a, Tensor* b, Tensor* grad, bool is_first);
  Tensor* pow_backwards(Tensor* a, float exp, Tensor* grad);
  Tensor* add_scalar_backwards(Tensor* grad);
  Tensor* sub_scalar_backwards(Tensor* grad, bool is_first);
  Tensor* mul_scalar_backwards(float scalar, Tensor* grad);
  Tensor* div_scalar_backwards(float scalar, Tensor* grad, bool is_first);
}

#endif  //!__BASIC_BACK__H__