#ifndef CPU_H
#define CPU_H

#include "tensor.h"

void add_tensor_cpu(Tensor* a, Tensor* b, float* out);
void sub_tensor_cpu(Tensor* a, Tensor* b, float* out);
void mul_tensor_cpu(Tensor* a, Tensor* b, float* out);
void div_tensor_cpu(Tensor* a, Tensor* b, float* out);
void add_broadcasted_tensor_cpu(Tensor* a, Tensor* b, float* out, int* broadcasted_shape, int broadcasted_size);
void sub_broadcasted_tensor_cpu(Tensor* a, Tensor* b, float* out, int* broadcasted_shape, int broadcasted_size);
void mul_broadcasted_tensor_cpu(Tensor* a, Tensor* b, float* out, int* broadcasted_shape, int broadcasted_size);

void scalar_mul_tensor_cpu(Tensor* a, float b, float* out);
void scalar_div_tensor_cpu(Tensor* a, float b, float* out);
void tensor_div_scalar_cpu(Tensor* a, float b, float* out);
void scalar_pow_tensor_cpu(float base, Tensor* a, float* out);
void tensor_pow_scalar_cpu(Tensor* a, float b, float* out);
void log_tensor_cpu(Tensor* a, float* out);
void sum_tensor_cpu(Tensor* a, float* out, int size, int* res_shape, int axis);
void max_tensor_cpu(Tensor* a, float* out, int size, int* res_shape, int axis);
void min_tensor_cpu(Tensor* a, float* out, int size, int* res_shape, int axis);
void equal_tensor_cpu(Tensor* a, Tensor* b, float* out);

#endif