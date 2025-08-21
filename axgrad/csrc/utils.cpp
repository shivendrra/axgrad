#include <stdio.h>
#include <stdlib.h>
#include "core/core.h"
#include "cpu/helpers.h"
#include "core/dtype.h"
#include "utils.h"

Tensor* zeros_like_tensor(Tensor* a) {
  float* out = (float*)malloc(a->size * sizeof(float));
  zeros_like_tensor_ops(out, a->size);
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, a->dtype);
  free(out);
  return result;
}

Tensor* zeros_tensor(int* shape, size_t size, size_t ndim, dtype_t dtype) {
  float* out = (float*)malloc(size * sizeof(float));
  zeros_tensor_ops(out, size);
  Tensor* result = create_tensor(out, ndim, shape, size, dtype);
  free(out);
  return result;
}

Tensor* ones_like_tensor(Tensor* a) {
  float* out = (float*)malloc(a->size * sizeof(float));
  ones_like_tensor_ops(out, a->size);
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, a->dtype);
  free(out);
  return result;
}

Tensor* ones_tensor(int* shape, size_t size, size_t ndim, dtype_t dtype) {
  float* out = (float*)malloc(size * sizeof(float));
  ones_tensor_ops(out, size);
  Tensor* result = create_tensor(out, ndim, shape, size, dtype);
  free(out);
  return result;
}

Tensor* randn_tensor(int* shape, size_t size, size_t ndim, dtype_t dtype) {
  float* out = (float*)malloc(size * sizeof(float));
  fill_randn(out, size);
  Tensor* result = create_tensor(out, ndim, shape, size, dtype);
  free(out);
  return result;
}

Tensor* randint_tensor(int low, int high, int* shape, size_t size, size_t ndim, dtype_t dtype) {
  float* out = (float*)malloc(size * sizeof(float));
  fill_randint(out, low, high, size);
  Tensor* result = create_tensor(out, ndim, shape, size, dtype);
  free(out);
  return result;
}

Tensor* uniform_tensor(int low, int high, int* shape, size_t size, size_t ndim, dtype_t dtype) {
  float* out = (float*)malloc(size * sizeof(float));
  fill_uniform(out, low, high, size);
  Tensor* result = create_tensor(out, ndim, shape, size, dtype);
  free(out);
  return result;
}

Tensor* fill_tensor(float fill_val, int* shape, size_t size, size_t ndim, dtype_t dtype) {
  float* out = (float*)malloc(size * sizeof(float));
  fill_tensor_ops(out, fill_val, size);
  Tensor* result = create_tensor(out, ndim, shape, size, dtype);
  free(out);
  return result;
}

Tensor* linspace_tensor(float start, float step, float end, int* shape, size_t size, size_t ndim, dtype_t dtype) {
  float* out = (float*)malloc(size * sizeof(float));
  float step_size = (size > 1) ? (end - start) / (size - 1) : 0.0f;
  linspace_tensor_ops(out, start, step_size, size);
  Tensor* result = create_tensor(out, ndim, shape, size, dtype);
  free(out);
  return result;
}

Tensor* arange_tensor(float start, float stop, float step, dtype_t dtype) {
  if (step == 0.0f) {
    fprintf(stderr, "Step cannot be zero\n");
    exit(EXIT_FAILURE);
  }
  size_t size = arange_size(start, stop, step);
  if (size == 0) {
    fprintf(stderr, "Invalid arange parameters\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(size * sizeof(float));
  arange_tensor_ops(out, start, stop, step, size);
  int* shape = (int*)malloc(sizeof(int));
  shape[0] = (int)size;
  Tensor* result = create_tensor(out, 1, shape, size, dtype);
  free(out);
  free(shape);
  return result;
}