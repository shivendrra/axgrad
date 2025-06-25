#ifndef __TENSOR_OPS__H__
#define __TENSOR_OPS__H__

#include <stdlib.h>
#include "core/core.h"

extern "C" {
  Tensor* matmul_tensor(Tensor* a, Tensor* b);
  Tensor* batch_matmul_tensor(Tensor* a, Tensor* b);
  Tensor* broadcasted_matmul_tensor(Tensor* a, Tensor* b);
  Tensor* dot_tensor(Tensor* a, Tensor* b);
  Tensor* batch_dot_tensor(Tensor* a, Tensor* b);
}

#endif