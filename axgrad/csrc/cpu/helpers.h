#ifndef __HELPER__H__
#define __HELPER__H__

extern "C" {
  void zeros_like_array_ops(float* out, size_t size);
  void zeros_array_ops(float* out, size_t size);
  void ones_like_array_ops(float* out, size_t size);
  void ones_array_ops(float* out, size_t size);

  void fill_array_ops(float* out, float value, size_t size);
  void linspace_array_ops(float* out, float start, float step_size, size_t size);
}

#endif  //!__HELPER__H__