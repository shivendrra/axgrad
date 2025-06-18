#ifndef __MATMUL_KERNEL_H__
#define __MATMUL_KERNEL_H__

#include <cuda_runtime.h>

__global__ void __matmul_kernel__(float* a, float* b, float* out, int rows1, int cols1, int cols2);
__host__ void matmul_tensor_cuda(float* a, float* b, float* out, int* shape1, int* shape2);

__global__ void __batched_matmul_kernel__(float* a, float* b, float* out, int batch_size, int rows1, int cols1, int cols2);
__host__ void batched_matmul_tensor_cuda(float* a, float* b, float* out, int* shape1, int* shape2);

__global__ void __broadcasted_batched_matmul_kernel__(float* a, float* b, float* out, int batch_size, int rows1, int cols1, int cols2);
__host__ void broadcasted_batched_matmul_tensor_cuda(float* a, float* b, float* out, int* shape1, int* shape2);

#endif