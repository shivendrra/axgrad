// statistical ops kernels

#ifndef __STATS_OPS_KERNEL_H__
#define __STATS_OPS_KERNEL_H__

#include <cuda_runtime.h>

// Mean operations
__global__ void __mean_kernel__(float* a, float* out, int size);
__global__ void __mean_axis_kernel__(float* a, float* out, int* strides, int* shape, int axis, int ndim, int axis_stride, int size, int result_size);
__host__ void mean_tensor_cuda(float* a, float* out, int axis, size_t size, int* strides, int* shape, size_t ndim);

// Variance operations
__global__ void __var_mean_kernel__(float* a, float* means, int size);
__global__ void __var_kernel__(float* a, float* means, float* out, int size);
__global__ void __var_axis_mean_kernel__(float* a, float* means, int* strides, int* shape, int axis, int ndim, int axis_stride, int size, int result_size);
__global__ void __var_axis_kernel__(float* a, float* means, float* out, int* strides, int* shape, int axis, int ndim, int axis_stride, int size, int result_size);
__host__ void var_tensor_cuda(float* a, float* out, int axis, size_t size, int* strides, int* shape, size_t ndim, int ddof);

// Standard deviation operations
__global__ void __std_finalize_kernel__(float* var, float* out, int size);
__host__ void std_tensor_cuda(float* a, float* out, int axis, size_t size, int* strides, int* shape, size_t ndim, int ddof);

#endif