#ifndef __CUDA_H__
#define __CUDA_H__

#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define TILE_SIZE 64

// core CUDA functions - work with float32 arrays directly
__host__ float* cpu_to_cuda(float* cpu_data, int device_id, size_t size);
__host__ float* cuda_to_cpu(float* gpu_data, size_t size);
__host__ void free_cuda(float* data);

// utility functions
__host__ int get_cuda_device_count(void);
__host__ void print_cuda_device_info(int device_id);

// contiguous ops
__device__ void flat_to_multi_index_cuda(size_t flat_idx, int* shape, size_t ndim, int* indices);
__device__ size_t calculate_flat_index_cuda(int* indices, int* strides, size_t ndim);
__global__ void contiguous_tensor_cuda(void* src_data, void* dst_data, int* strides, int* shape, size_t ndim, size_t elem_size, size_t total_size);

__global__ void assign_tensor_cudtensor_kernel(float* a, float* out, int size);
__host__ void assign_tensor_cuda(float* a, float* out, size_t size);

#endif