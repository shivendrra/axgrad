#ifndef __UNARY_OPS_KERNEL_H__
#define __UNARY_OPS_KERNEL_H__

#include <cuda_runtime.h>

__global__ void __log_kernel__(float* a, float* out, size_t size);
__host__ void log_tensor_cuda(float* a, float* out, size_t size);

__global__ void __exp_kernel__(float* a, float* out, size_t size);
__host__ void exp_tensor_cuda(float* a, float* out, size_t size);

__global__ void __abs_kernel__(float* a, float* out, size_t size);
__host__ void abs_tensor_cuda(float* a, float* out, size_t size);

__global__ void __relu_kernel__(float* a, float* out, size_t size);
__host__ void relu_tensor_cuda(float* a, float* out, size_t size);

__global__ void __sigmoid_kernel__(float* a, float* out, size_t size);
__host__ void sigmoid_tensor_cuda(float* a, float* out, size_t size);

__global__ void __gelu_kernel__(float* a, float* out, size_t size);
__host__ void gelu_tensor_cuda(float* a, float* out, size_t size);

__global__ void __sin_kernel__(float* a, float* out, size_t size);
__host__ void sin_tensor_cuda(float* a, float* out, size_t size);

__global__ void __cos_kernel__(float* a, float* out, size_t size);
__host__ void cos_tensor_cuda(float* a, float* out, size_t size);

__global__ void __tan_kernel__(float* a, float* out, size_t size);
__host__ void tan_tensor_cuda(float* a, float* out, size_t size);

__global__ void __sinh_kernel__(float* a, float* out, size_t size);
__host__ void sinh_tensor_cuda(float* a, float* out, size_t size);

__global__ void __cosh_kernel__(float* a, float* out, size_t size);
__host__ void cosh_tensor_cuda(float* a, float* out, size_t size);

__global__ void __tanh_kernel__(float* a, float* out, size_t size);
__host__ void tanh_tensor_cuda(float* a, float* out, size_t size);