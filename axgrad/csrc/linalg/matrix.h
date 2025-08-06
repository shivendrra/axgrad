#ifndef __MATRIX__H__
#define __MATRIX__H__

#include "../core/core.h"
#include "../core/dtype.h"

extern "C" {
  Tensor* det_tensor(Tensor* a);
  Tensor* batched_det_tensor(Tensor* a);
  Tensor* inv_tensor(Tensor* a);
  Tensor* solve_tensor(Tensor* a, Tensor* b);
  Tensor* lstsq_tensor(Tensor* a, Tensor* b);
}

#endif  //!__MATRIX__H__