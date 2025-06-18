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

#endif