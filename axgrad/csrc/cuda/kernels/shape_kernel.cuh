#ifndef __SHAPE_KERNEL_H__
#define __SHAPE_KERNEL_H__

#include <cuda_runtime.h>

__global__ void __transpose_1D_kernel__(float* a, float* out, size_t size);
__host__ void transpose_1D_tensor_cuda(float* a, float* out, size_t size);

__global__ void __transpose_2D_kernel__(float* a, float* out, int rows, int cols);
__host__ void transpose_2D_tensor_cuda(float* a, float* out, int* shape);

__global__ void __transpose_3D_kernel__(float* a, float* out, int batch, int rows, int cols);
__host__ void transpose_3D_tensor_cuda(float* a, float* out, int* shape);

#endif