#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <cuda_runtime.h>
#include "cuda.cuh"

// CUDA error checking macro
#define CUDA_CHECK(call) \
  do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

__host__ float* cpu_to_cuda(float* cpu_data, int device_id, size_t size) {
  if (cpu_data == NULL) {
    fprintf(stderr, "CPU data pointer is null!\n");
    exit(EXIT_FAILURE);
  }

  // setting the specified GPU device
  CUDA_CHECK(cudaSetDevice(device_id));

  // allocating GPU memory for float32 data
  float* gpu_float_data;
  size_t data_size = size * sizeof(float);
  CUDA_CHECK(cudaMalloc((void**)&gpu_float_data, data_size));

  // copying float32 data from CPU to GPU
  CUDA_CHECK(cudaMemcpy(gpu_float_data, cpu_data, data_size, cudaMemcpyHostToDevice));
  free(cpu_data);  // cleaning up temporary CPU float data
  // printf("float data moved to GPU device %d as float32 array\n", device_id);
  return gpu_float_data;
}

__host__ float* cuda_to_cpu(float* gpu_data, size_t size) {
  if (gpu_data == NULL || !size) {
    fprintf(stderr, "Invalid input parameters!\n");
    exit(EXIT_FAILURE);
  }

  // checking if data is actually on GPU
  cudaPointerAttributes attributes;
  cudaError_t error = cudaPointerGetAttributes(&attributes, gpu_data);
  if (error != cudaSuccess || attributes.type != cudaMemoryTypeDevice) {
    fprintf(stderr, "Data is not on GPU or invalid pointer!\n");
    exit(EXIT_FAILURE);
  }

  // allocating CPU memory for float32 data
  float* cpu_data = (float*)malloc(size * sizeof(float));
  if (cpu_data == NULL) {
    fprintf(stderr, "Memory allocation failed for CPU float data!\n");
    exit(EXIT_FAILURE);
  }

  // copying float32 data from GPU to CPU
  CUDA_CHECK(cudaMemcpy(cpu_data, gpu_data, size * sizeof(float), cudaMemcpyDeviceToHost));
  // printf("GPU float32 data converted to CPU\n");
  return cpu_data;
}

__host__ void free_cuda(float* data) {
  if (data == NULL) {
    fprintf(stderr, "Cannot free null GPU data pointer!\n");
    return;
  }

  // checking if the pointer is actually on GPU
  cudaPointerAttributes attributes;
  cudaError_t error = cudaPointerGetAttributes(&attributes, data);
  if (error != cudaSuccess) {
    fprintf(stderr, "Invalid GPU pointer or error checking pointer attributes!\n");
    return;
  }

  if (attributes.type != cudaMemoryTypeDevice) {
    fprintf(stderr, "Pointer is not GPU memory!\n");
    return;
  }

  // freeing the GPU memory
  CUDA_CHECK(cudaFree(data));
  printf("GPU memory freed\n");
}

// utility function to get GPU device count
__host__ int get_cuda_device_count(void) {
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  if (error != cudaSuccess) {
    fprintf(stderr, "Failed to get CUDA device count: %s\n", cudaGetErrorString(error));
    return 0;
  }
  return device_count;
}

// utility function to print GPU device information
__host__ void print_cuda_device_info(int device_id) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
  
  printf("GPU Device %d: %s\n", device_id, prop.name);
  printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
  printf("  Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
  printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
  printf("  Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
  printf("  Warp Size: %d\n", prop.warpSize);
}

__device__ size_t calculate_flat_index_cuda(int* indices, int* strides, size_t ndim) {
  size_t flat_idx = 0;
  for (size_t i = 0; i < ndim; i++) {
    flat_idx += indices[i] * strides[i];
  }
  return flat_idx;
}

__device__ void flat_to_multi_index_cuda(size_t flat_idx, int* shape, size_t ndim, int* indices) {
  for (int i = ndim - 1; i >= 0; i--) {
    indices[i] = flat_idx % shape[i];
    flat_idx /= shape[i];
  }
}

__global__ void contiguous_tensor_cuda(void* src_data, void* dst_data, int* strides, int* shape, size_t ndim, size_t elem_size, size_t total_size) {
  size_t flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (flat_idx >= total_size) return;

  extern __shared__ int shared[];
  int* indices = shared + threadIdx.x * ndim;

  flat_to_multi_index_cuda(flat_idx, shape, ndim, indices);
  size_t src_offset = 0;
  for (size_t dim = 0; dim < ndim; dim++) {
    src_offset += indices[dim] * strides[dim] * elem_size;
  }

  char* src = (char*)src_data;
  char* dst = (char*)dst_data;
  size_t dst_offset = flat_idx * elem_size;
  for (size_t i = 0; i < elem_size; i++) {
    dst[dst_offset + i] = src[src_offset + i];
  }
}

__global__ void __assign_tensor_kernel__(float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = a[i];
  }
}

__host__ void assign_tensor_cuda(float* a, float* out, size_t size) {
  int n_of_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  __assign_tensor_kernel__<<<n_of_blocks, THREADS_PER_BLOCK>>>(a, out, size);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}