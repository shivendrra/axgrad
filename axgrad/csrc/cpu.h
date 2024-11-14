#ifndef CPU_H
#define CPU_H

#include "tensor.h"

void add_tensor_cpu(Tensor* a, Tensor* b, float* out);
void sub_tensor_cpu(Tensor* a, Tensor* b, float* out);
void mul_tensor_cpu(Tensor* a, Tensor* b, float* out);
void add_broadcasted_tensor_cpu(Tensor* a, Tensor* b, float* out, int* broadcasted_shape, int broadcasted_size);
void sub_broadcasted_tensor_cpu(Tensor* a, Tensor* b, float* out, int* broadcasted_shape, int broadcasted_size);
void mul_broadcasted_tensor_cpu(Tensor* a, Tensor* b, float* out, int* broadcasted_shape, int broadcasted_size);

#endif