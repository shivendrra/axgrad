#include <math.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include "matmul_kernel.cuh"
#include "../cuda.cuh"

__global__ void __matmul_kernel__(float* a, float* b, float* out, int rows1, int cols1, int cols2) {
  // shared memory for tiles
  __shared__ float tile1[TILE_SIZE][TILE_SIZE];
  __shared__ float tile2[TILE_SIZE][TILE_SIZE];

  // thread indices
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // output position
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float sum = 0.0;
  // iterate over tiles
  for (int i = 0; i < (cols1 + TILE_SIZE - 1) / TILE_SIZE; ++i) {
    if (row < rows1 && i * TILE_SIZE + tx < cols1)
      tile1[ty][tx] = a[row * cols1 + i * TILE_SIZE + tx];
    else
      tile1[ty][tx] = 0.0;
    if (col < cols2 && i * TILE_SIZE + ty < cols1)
      tile2[ty][tx] = b[(i * TILE_SIZE + ty) * cols2 + col];
    else
      tile2[ty][tx] = 0.0;
    __syncthreads();
    for (int k = 0; k < TILE_SIZE; ++k)
      sum += tile1[ty][k] * tile2[k][tx];
    __syncthreads();
  }
  if (row < rows1 && col < cols2)
    out[row * cols2 + col] = sum;
}

__host__ void matmul_tensor_cuda(float* a, float* b, float* out, int* shape1, int* shape2) {
  int rows1 = shape1[0], cols1 = shape1[1];
  int rows2 = shape2[0], cols2 = shape2[1];

  dim3 threadsPerBlock(16, 16);
  dim3 n_of_blocks((cols2 + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows1 + threadsPerBlock.y - 1) / threadsPerBlock.y);
  __matmul_kernel__<<<n_of_blocks, threadsPerBlock>>>(a, b, out, rows1, cols1, cols2);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __batched_matmul_kernel__(float* a, float* b, float* out, int batch_size, int rows1, int cols1, int cols2) {
  int batch = blockIdx.z;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (batch < batch_size && row < rows1 && col < cols2) {
    float sum = 0.0f;
    for (int k = 0; k < cols1; ++k) {
      sum += a[batch * rows1 * cols1 + row * cols1 + k] * b[batch * cols1 * cols2 + k * cols2 + col];
    }
    out[batch * rows1 * cols2 + row * cols2 + col] = sum;
  }    
}

__host__ void batched_matmul_tensor_cuda(float* a, float* b, float* out, int* shape1, int* shape2) {
  int batch_size = shape1[0];
  int rows1 = shape1[1];
  int cols1 = shape1[2];
  int cols2 = shape2[2];

  dim3 threadsPerBlock(16, 16);
  dim3 n_of_blocks((cols2 + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows1 + threadsPerBlock.y - 1) / threadsPerBlock.y, batch_size);
  __batched_matmul_kernel__<<<n_of_blocks, threadsPerBlock>>>(a, b, out, batch_size, rows1, cols1, cols2);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  cudaDeviceSynchronize();
}

__global__ void __broadcasted_batched_matmul_kernel__(float* a, float* b, float* out, int batch_size, int rows1, int cols1, int cols2) {
  int batch = blockIdx.z;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (batch < batch_size && row < rows1 && col < cols2) {
    float sum = 0.0f;
    for (int k = 0; k < cols1; ++k) {
      sum += a[row * cols1 + k] * b[batch * cols1 * cols2 + k * cols2 + col];
    }
    out[batch * rows1 * cols2 + row * cols2 + col] = sum;
  }    
}

__host__ void broadcasted_batched_matmul_tensor_cuda(float* a, float* b, float* out, int* shape1, int* shape2) {
  int batch_size = shape2[0];
  int rows1 = shape1[0];
  int cols1 = shape1[1];
  int cols2 = shape2[2];

  dim3 threadsPerBlock(16, 16);
  dim3 n_of_blocks((cols2 + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows1 + threadsPerBlock.y - 1) / threadsPerBlock.y, batch_size);
  __broadcasted_batched_matmul_kernel__<<<n_of_blocks, threadsPerBlock>>>(a, b, out, batch_size, rows1, cols1, cols2);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  cudaDeviceSynchronize();
}