#ifndef __BINARY_OPS_KERNEL_H__
#define __BINARY_OPS_KERNEL_H__

#include <cuda_runtime.h>

__global__ void __add_kernel__(float* a, float* b, float* out, size_t size);
__host__ void add_tensor_cuda(float* a, float* b, float* out, size_t size);

__global__ void __add_scalar_kernel__(float* a, float b, float* out, size_t size);
__host__ void add_scalar_tensor_cuda(float* a, float b, float* out, size_t size);

__global__ void __sub_kernel__(float* a, float* b, float* out, size_t size);
__host__ void sub_tensor_cuda(float* a, float* b, float* out, size_t size);

__global__ void __sub_scalar_kernel__(float* a, float b, float* out, size_t size);
__host__ void sub_scalar_tensor_cuda(float* a, float b, float* out, size_t size);

__global__ void __mul_kernel__(float* a, float* b, float* out, size_t size);
__host__ void mul_tensor_cuda(float* a, float* b, float* out, size_t size);

__global__ void __mul_scalar_kernel__(float* a, float b, float* out, size_t size);
__host__ void mul_scalar_tensor_cuda(float* a, float b, float* out, size_t size);

__global__ void __div_kernel__(float* a, float* b, float* out, size_t size);
__host__ void div_tensor_cuda(float* a, float* b, float* out, size_t size);

__global__ void __div_scalar_kernel__(float* a, float b, float* out, size_t size);
__host__ void div_scalar_tensor_cuda(float* a, float b, float* out, size_t size);

__global__ void __pow_tensor_scalar_kernel__(float* a, float exp, float* out, size_t size);
__host__ void pow_tensor_scalar_cuda(float* a, float exp, float* out, size_t size);

__global__ void __pow_scalar_tensor_kernel__(float base, float* exp, float* out, size_t size);
__host__ void pow_scalar_tensor_cuda(float base, float* exp, float* out, size_t size);

__global__ void __add_broadcasted_kernel__(float* a, float* b, float* out, int* a_strides, int* b_strides, int* broadcasted_shape, int ndim, size_t size);
__host__ void add_broadcasted_tensor_cuda(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape);

__global__ void __sub_broadcasted_kernel__(float* a, float* b, float* out, int* a_strides, int* b_strides, int* broadcasted_shape, int ndim, size_t size);
__host__ void sub_broadcasted_tensor_cuda(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape);

__global__ void __mul_broadcasted_kernel__(float* a, float* b, float* out, int* a_strides, int* b_strides, int* broadcasted_shape, int ndim, size_t size);
__host__ void mul_broadcasted_tensor_cuda(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape);

__global__ void __div_broadcasted_kernel__(float* a, float* b, float* out, int* a_strides, int* b_strides, int* broadcasted_shape, int ndim, size_t size);
__host__ void div_broadcasted_tensor_cuda(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape);

#endif