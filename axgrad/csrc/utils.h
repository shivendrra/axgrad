#ifndef __UTILS__H__
#define __UTILS__H__

#include "core/core.h"
#include "core/dtype.h"

extern "C" {
  Tensor* zeros_like_tensor(Tensor* a);
  Tensor* ones_like_tensor(Tensor* a);
  Tensor* zeros_tensor(int* shape, size_t size, size_t ndim, dtype_t dtype);
  Tensor* ones_tensor(int* shape, size_t size, size_t ndim, dtype_t dtype);
  Tensor* randn_tensor(int* shape, size_t size, size_t ndim, dtype_t dtype);
  Tensor* randint_tensor(int low, int high, int* shape, size_t size, size_t ndim, dtype_t dtype);
  Tensor* uniform_tensor(int low, int high, int* shape, size_t size, size_t ndim, dtype_t dtype);
  Tensor* fill_tensor(float fill_val, int* shape, size_t size, size_t ndim, dtype_t dtype);
  Tensor* linspace_tensor(float start, float step, float end, int* shape, size_t size, size_t ndim, dtype_t dtype);
  Tensor* arange_tensor(float start, float stop, float step, dtype_t dtype);
}

#endif  //!__UTILS__H__