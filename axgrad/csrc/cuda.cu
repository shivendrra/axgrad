#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "tensor.h"

#define THREADS_PER_BLOCK 128
#define TILE_SIZE 32

__host__ void cpu_to_cuda(Tensor* a, int device_id) {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (device_id >= deviceCount) {
    fprintf(stderr, "Could not send tensor to device %d, only %d devices available\n", device_id, deviceCount);
    exit(1);
  }
  cudaSetDevice(device_id);

  float* data_tmp;
  cudaMalloc((void**)&data_tmp, a->size * sizeof(float));
  cudaMemcpy(data_tmp, a->data, a->size * sizeof(float), cudaMemcpyHostToDevice);
  a->data = data_tmp;
  a->device = (char*)malloc(strlen("cuda") + 1);
  strcpy(a->device, "cuda");
}

__host__ void cuda_to_cpu(Tensor* a) {
  float* data_tmp = (float*)malloc(a->size * sizeof(float));
  cudaMemcpy(data_tmp, a->data, a->size * sizeof(float), cudaMemcpyHostToDevice);
  cudaFree(a->data);
  a->data = data_tmp;
  a->deivce = (char*)malloc(strlen("cpu") + 1);
  strcpy(a->device, "cpu");
}

__host__ void free_cuda(float* data) {
  cudaFree(data);
}

__global__ void add_tensor_cuda_kernel(float* a, float* b, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = a[i] + b[i];
  }
}

__host__ void add_tensor_cuda(Tensor* a, Tensor* b, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  add_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->data, b->data, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void sub_tensor_cuda_kernel(float* a, float* b, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = a[i] - b[i];
  }
}

__host__ void sub_tensor_cuda(Tensor* a, Tensor* b, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  sub_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->data, b->data, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void mul_tensor_cuda_kernel(float* a, float* b, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = a[i] * b[i];
  }
}

__host__ void mul_tensor_cuda(Tensor* a, Tensor* b, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  mul_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->data, b->data, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void div_tensor_cuda_kernel(float* a, float* b, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = a[i] / b[i];
  }
}

__host__ void div_tensor_cuda(Tensor* a, Tensor* b, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  div_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->data, b->data, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}