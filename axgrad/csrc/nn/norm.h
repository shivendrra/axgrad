#ifndef __NORM__H__
#define __NORM__H__

#include "../core/core.h"
#include "../core/dtype.h"

extern "C" {
  Tensor* clip_tensor(Tensor* a, float max_val);
  Tensor* clamp_tensor(Tensor* a, float min_val, float max_val);
  Tensor* mm_norm_tensor(Tensor* a);
  Tensor* std_norm_tensor(Tensor* a);
  Tensor* rms_norm_tensor(Tensor* a);
  Tensor* l1_norm_tensor(Tensor* a);
  Tensor* l2_norm_tensor(Tensor* a);
  Tensor* unit_norm_tensor(Tensor* a);
  Tensor* robust_norm_tensor(Tensor* a);
}

#endif  //!__NORM__H__