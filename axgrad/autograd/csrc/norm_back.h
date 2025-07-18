#ifndef __NORM_BACK__H__
#define __NORM_BACK__H__

#include "../../csrc/core/core.h"
#include "../../csrc/core/dtype.h"

extern "C" {
  Tensor* clip_backwards(Tensor* a, Tensor* grad, float max_val);
  Tensor* clamp_backwards(Tensor* a, Tensor* grad, float min_val, float max_val);
  Tensor* mm_norm_backwards(Tensor* a, Tensor* grad);
  Tensor* std_norm_backwards(Tensor* a, Tensor* grad);
  Tensor* rms_norm_backwards(Tensor* a, Tensor* grad);
  Tensor* l1_norm_backwards(Tensor* a, Tensor* grad);
  Tensor* l2_norm_backwards(Tensor* a, Tensor* grad);
  Tensor* unit_norm_backwards(Tensor* a, Tensor* grad);
  Tensor* robust_norm_backwards(Tensor* a, Tensor* grad);
}

#endif  //!__NORM_BACK__H__