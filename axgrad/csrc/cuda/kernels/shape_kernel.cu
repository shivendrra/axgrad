#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "shape_kernel.cuh"
#include "../cuda.cuh"

__global__ void __transpose_1D_kernel__(float* a, float* out, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = a[i];
  }
}

__host__ void transpose_1D_tensor_cuda(float* a, float* out, size_t size) {
  int n_of_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  __transpose_1D_kernel__<<<n_of_blocks, THREADS_PER_BLOCK>>>(a, out, size);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __transpose_2D_kernel__(float* a, float* out, int rows, int cols) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < rows && j < cols) {
    out[j * rows + i] = a[i * cols + j];
  }
}

__host__ void transpose_2D_tensor_cuda(float* a, float* out, int* shape) {
  int rows = shape[0], cols = shape[1];

  dim3 threadsPerBlock(16, 16);
  dim3 n_of_blocks((rows + threadsPerBlock.x - 1) / threadsPerBlock.x, (cols + threadsPerBlock.y - 1) / threadsPerBlock.y);
  __transpose_2D_kernel__<<<n_of_blocks, threadsPerBlock>>>(a, out, rows, cols);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __transpose_3D_kernel__(float* a, float* out, int batch, int rows, int cols) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < batch && j < rows && k < cols) {
    out[k * rows * batch + j * batch + i] = a[i * rows * cols + j * cols + k];
  }
}

__host__ void transpose_3D_tensor_cuda(float* a, float* out, int* shape) {
  int batch = shape[0], rows = shape[1], cols = shape[2];

  dim3 threadsPerBlock(8, 8, 8);
  dim3 n_of_blocks((batch + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y, (cols + threadsPerBlock.z - 1) / threadsPerBlock.z);
  __transpose_3D_kernel__<<<n_of_blocks, threadsPerBlock>>>(a, out, batch, rows, cols);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}