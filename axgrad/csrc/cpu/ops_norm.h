#ifndef __OPS_NORM__H__
#define __OPS_NORM__H__

#include <stddef.h>

extern "C" {
  void clip_tensor_ops(float* a, float* out, float max_val, size_t size);
  void clamp_tensor_ops(float* a, float* out, float min_val, float max_val, size_t size);
  void mm_norm_tensor_ops(float* a, float* out, size_t size);
  void std_norm_tensor_ops(float* a, float* out, size_t size);
  void rms_norm_tensor_ops(float* a, float* out, size_t size);
  void l1_norm_tensor_ops(float* a, float* out, size_t size);
  void l2_norm_tensor_ops(float* a, float* out, size_t size);
  void unit_norm_tensor_ops(float* a, float* out, size_t size);
  void robust_norm_tensor_ops(float* a, float* out, size_t size);
}

#endif  //!__OPS_NORM__H__