#ifndef __NORM__H__
#define __NORM__H__

#include <stddef.h>

extern "C" {
  void clip_backwards_ops(float* a, float* grad, float* out, float max_val, size_t size);
  void clamp_backwards_ops(float* a, float* grad, float* out, float min_val, float max_val, size_t size);
  void mm_norm_backwards_ops(float* a, float* grad, float* out, size_t size);
  void std_norm_backwards_ops(float* a, float* grad, float* out, size_t size);
  void rms_norm_backwards_ops(float* a, float* grad, float* out, size_t size);
  void l1_norm_backwards_ops(float* a, float* grad, float* out, size_t size);
  void l2_norm_backwards_ops(float* a, float* grad, float* out, size_t size);
  void unit_norm_backwards_ops(float* a, float* grad, float* out, size_t size);
  void robust_norm_backwards_ops(float* a, float* grad, float* out, size_t size);
}

#endif  //!__NORM__H__