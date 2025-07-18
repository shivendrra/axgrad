#ifndef __VECTOR__H__
#define __VECTOR__H__

#include "../core/core.h"
#include "../core/dtype.h"

extern "C" {
  Tensor* vector_dot(Tensor* a, Tensor* b);
  Tensor* vector_matrix_dot(Tensor* vec, Tensor* mat);
  Tensor* vector_inner(Tensor* a, Tensor* b);
  Tensor* vector_outer(Tensor* a, Tensor* b);
  Tensor* vector_cross(Tensor* a, Tensor* b);
  Tensor* vector_cross_axis(Tensor* a, Tensor* b, int axis);
}

#endif  //!__VECTOR__H__