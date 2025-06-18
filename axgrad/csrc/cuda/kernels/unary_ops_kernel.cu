#include <math.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include "unary_ops_kernel.cuh"
#include "../cuda.cuh"

__global__ void __log_kernel_(float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = logf(a[i]);
  }
}

__host__ void log_tensor_cuda(float* a, float* out, size_t size) {
  int n_of_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  __log_kernel__<<<n_of_blocks, THREADS_PER_BLOCK>>>(a, out, size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __exp_kernel_(float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = expf(a[i]);
  }
}

__host__ void exp_tensor_cuda(float* a, float* out, size_t size) {
  int n_of_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  __exp_kernel__<<<n_of_blocks, THREADS_PER_BLOCK>>>(a, out, size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __abs_kernel_(float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = fabsf(a[i]);
  }
}

__host__ void abs_tensor_cuda(float* a, float* out, size_t size) {
  int n_of_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  __abs_kernel__<<<n_of_blocks, THREADS_PER_BLOCK>>>(a, out, size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __relu_kernel_(float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = fmax(a[i], 0.0);
  }
}

__host__ void relu_tensor_cuda(float* a, float* out, size_t size) {
  int n_of_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  __relu_kernel__<<<n_of_blocks, THREADS_PER_BLOCK>>>(a, out, size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __sigmoid_kernel_(float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = 1.0f / (1.0f + expf(-a[i]));
  }
}

__host__ void sigmoid_tensor_cuda(float* a, float* out, size_t size) {
  int n_of_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  __sigmoid_kernel__<<<n_of_blocks, THREADS_PER_BLOCK>>>(a, out, size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __gelu_kernel__(float* a, float* out, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = 0.5 * a[i] * (1.0 + tanhf(0.79788456 * (a[i] + 0.044715 * a[i] * a[i] * a[i])));
  }
}

__host__ void gelu_tensor_cuda(float* a, float* out, size_t size) {
  int n_of_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  __gelu_kernel__<<<n_of_blocks, THREADS_PER_BLOCK>>>(a, out, size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __sin_kernel_(float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = sinf(a[i], 0.0);
  }
}

__host__ void sin_tensor_cuda(float* a, float* out, size_t size) {
  int n_of_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  __sin_kernel__<<<n_of_blocks, THREADS_PER_BLOCK>>>(a, out, size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __cos_kernel_(float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = cosf(a[i], 0.0);
  }
}

__host__ void cos_tensor_cuda(float* a, float* out, size_t size) {
  int n_of_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  __cos_kernel__<<<n_of_blocks, THREADS_PER_BLOCK>>>(a, out, size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __tan_kernel_(float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = tanf(a[i], 0.0);
  }
}

__host__ void tan_tensor_cuda(float* a, float* out, size_t size) {
  int n_of_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  __tan_kernel__<<<n_of_blocks, THREADS_PER_BLOCK>>>(a, out, size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __sinh_kernel_(float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = sinh(a[i], 0.0);
  }
}

__host__ void sinh_tensor_cuda(float* a, float* out, size_t size) {
  int n_of_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  __sinh_kernel__<<<n_of_blocks, THREADS_PER_BLOCK>>>(a, out, size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __cosh_kernel_(float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = coshf(a[i], 0.0);
  }
}

__host__ void cosh_tensor_cuda(float* a, float* out, size_t size) {
  int n_of_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  __cosh_kernel__<<<n_of_blocks, THREADS_PER_BLOCK>>>(a, out, size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __tanh_kernel_(float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = tanhf(a[i], 0.0);
  }
}

__host__ void tanh_tensor_cuda(float* a, float* out, size_t size) {
  int n_of_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  __tanh_kernel__<<<n_of_blocks, THREADS_PER_BLOCK>>>(a, out, size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}