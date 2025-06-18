// reduction ops kernels

#ifndef __RED_OPS_KERNEL_H__
#define __RED_OPS_KERNEL_H__

#include <cuda_runtime.h>

__global__ void __sum_kernel__(float* a, float* out, int size);
__global__ void __sum_axis_kernel__(float* a, float* out, int* strides, int* shape, int axis, int ndim, int axis_stride, int size, int result_size);
__host__ void sum_tensor_cuda(float* a, float* out, int axis, size_t size, int* strides, int* shape, size_t ndim);

__global__ void __max_kernel__(float* a, float* out, int size);
__device__ float __atomicMaxFloat__(float* address, float val);
__global__ void __max_axis_kernel__(float* a, float* out, int* strides, int* shape, int axis, int ndim, int axis_stride, int size, int result_size);
__host__ void max_tensor_cuda(float* a, float* out, int axis, size_t size, int* strides, int* shape, size_t ndim);

__global__ void __min_kernel__(float* a, float* out, int size);
__device__ float __atomicMinFloat__(float* address, float val);
__global__ void __min_axis_kernel__(float* a, float* out, int* strides, int* shape, int axis, int ndim, int axis_stride, int size, int result_size);
__host__ void min_tensor_cuda(float* a, float* out, int axis, size_t size, int* strides, int* shape, size_t ndim);

#endif