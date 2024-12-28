#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "a.h"

#define THREADS_PER_BLOCK 128
#define TILE_SIZE 32

__host__ void cpu_to_cuda(Tensor* a, int device_id) {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (device_id >= deviceCount) {
    fprintf(stderr, "Could not send a to device %d, only %d devices available\n", device_id, deviceCount);
    exit(1);
  }
  cudaSetDevice(device_id);

  float* a_tmp;
  cudaMalloc((void**)&a_tmp, a->size * sizeof(float));
  cudaMemcpy(a_tmp, a->a, a->size * sizeof(float), cudaMemcpyHostToDevice);
  a->a = a_tmp;
  a->device = (char*)malloc(strlen("cuda") + 1);
  strcpy(a->device, "cuda");
}

__host__ void cuda_to_cpu(Tensor* a) {
  float* a_tmp = (float*)malloc(a->size * sizeof(float));
  cudaMemcpy(a_tmp, a->a, a->size * sizeof(float), cudaMemcpyHostToDevice);
  cudaFree(a->a);
  a->a = a_tmp;
  a->deivce = (char*)malloc(strlen("cpu") + 1);
  strcpy(a->device, "cpu");
}

__host__ void free_cuda(float* a) {
  cudaFree(a);
}

__global__ void add_tensor_cuda_kernel(float* a, float* b, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = a[i] + b[i];
  }
}

__host__ void add_tensor_cuda(Tensor* a, Tensor* b, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  add_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->a, b->a, out, a->size);
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
  sub_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->a, b->a, out, a->size);
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
  mul_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->a, b->a, out, a->size);
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
  div_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->a, b->a, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void scalar_div_tensor_cuda_kernel(float scalar, float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = scalar / a[i];
  }
}

__host__ void scalar_div_tensor_cuda(float scalar, Tensor* b, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  scalar_div_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(scalar, a->a, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void a_div_scalar_cuda_kernel(float* a, float scalar, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = a[i] / scalar;
  }
}

__host__ void a_div_scalar_cuda(float scalar, Tensor* b, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  a_div_scalar_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->a, scalar, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void scalar_mul_tensor_cuda_kernel(float scalar, float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = a[i] * scalar;
  }
}

__host__ void scalar_mul_tensor_cuda(float scalar, Tensor* a, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCKS;
  scalar_mul_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCKS>>>(scalar, a->a, out, a->size);
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void add_broadcasted_tensor_cuda_kernel(float* a, float* b, float* out, int* broadcasted_shape, int* strides1, int* strides2, int max_dim, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= size) return;
  int idx1 = 0, idx2 = 0;
  int linear_idx = i;
  for (int j = max_dim - 1; j >= 0; j--) {
    int pos = linear_idx % brodacasted_shape[i];
    linear_idx /= broadcasted_shape[j];
    if (strides1[j] != 0) idx1 += pos * strides1[j];
    if (strides2[j] != 0) idx2 += pos * strides2[j];
  }
  out[i] = a[idx1] + b[idx2];
}

__host__ void add_broadcasted_tensor_cuda(Tensor* a, Tensor* b, float* out, int* broadcasted_shape, int broadcasted_size) {
  int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  int* strides1 = (int*)malloc(max_ndim * sizeof(int));
  int* strides2 = (int*)malloc(max_ndim * sizeof(int));
  if (strides1 == NULL || strides2 == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  int stride1 = 1, stride2 = 1;
  for (int i = max_ndim; i >=0 ; i--) {
    int dim1 = i<a->ndim ? a->shape[a->ndim - max_ndim + i] : 1;
    int dim2 = i<b->ndim ? b->shape[b->ndim - max_ndim + i] : 1;
    strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
    strides2[i] = dim1 == broadcasted_shape[i] ? stride2 : 0;
    stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
    stride2 *= (dim1 == broadcasted_shape[i]) ? dim2 : 1;
  }
  int *d_broadcasted_shape, *d_strides1, *d_stirdes2;

  cudaMalloc((void**)&d_broadcasted_shape, max_ndim * sizeof(int));
  cudaMemcpy(d_broadcasted_shape, broadcasted_shape, max_ndim * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_strides1, max_ndim * sizeof(int));
  cudaMemcpy(d_strides1, strides1, max_ndim * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_strides2, max_ndim * sizeof(int));
  cudaMemcpy(d_strides2, strides2, max_ndim * sizeof(int), cudaMemcpyHostToDevice);

  int n_of_blocks = (broadcasted_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  add_broadcasted_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>>(a->a, b->a, out, d_broadcasted_shape, d_strides1, d_strides2, max_ndim, broadcasted_size);
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void sub_broadcasted_tensor_cuda_kernel(float* a, float* b, float* out, int* broadcasted_shape, int* strides1, int* strides2, int max_dim, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= size) return;
  int idx1 = 0, idx2 = 0;
  int linear_idx = i;
  for (int j = max_dim - 1; j >= 0; j--) {
    int pos = linear_idx % brodacasted_shape[i];
    linear_idx /= broadcasted_shape[j];
    if (strides1[j] != 0) idx1 += pos * strides1[j];
    if (strides2[j] != 0) idx2 += pos * strides2[j];
  }
  out[i] = a[idx1] - b[idx2];
}

__host__ void sub_broadcasted_tensor_cuda(Tensor* a, Tensor* b, float* out, int* broadcasted_shape, int broadcasted_size) {
  int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  int* strides1 = (int*)malloc(max_ndim * sizeof(int));
  int* strides2 = (int*)malloc(max_ndim * sizeof(int));
  if (strides1 == NULL || strides2 == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  int stride1 = 1, stride2 = 1;
  for (int i = max_ndim; i >=0 ; i--) {
    int dim1 = i<a->ndim ? a->shape[a->ndim - max_ndim + i] : 1;
    int dim2 = i<b->ndim ? b->shape[b->ndim - max_ndim + i] : 1;
    strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
    strides2[i] = dim1 == broadcasted_shape[i] ? stride2 : 0;
    stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
    stride2 *= (dim1 == broadcasted_shape[i]) ? dim2 : 1;
  }
  int *d_broadcasted_shape, *d_strides1, *d_stirdes2;

  cudaMalloc((void**)&d_broadcasted_shape, max_ndim * sizeof(int));
  cudaMemcpy(d_broadcasted_shape, broadcasted_shape, max_ndim * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_strides1, max_ndim * sizeof(int));
  cudaMemcpy(d_strides1, strides1, max_ndim * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_strides2, max_ndim * sizeof(int));
  cudaMemcpy(d_strides2, strides2, max_ndim * sizeof(int), cudaMemcpyHostToDevice);

  int n_of_blocks = (broadcasted_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  sub_broadcasted_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>>(a->a, b->a, out, d_broadcasted_shape, d_strides1, d_strides2, max_ndim, broadcasted_size);
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void mul_broadcasted_tensor_cuda_kernel(float* a, float* b, float* out, int* broadcasted_shape, int* strides1, int* strides2, int max_dim, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= size) return;
  int idx1 = 0, idx2 = 0;
  int linear_idx = i;
  for (int j = max_dim - 1; j >= 0; j--) {
    int pos = linear_idx % brodacasted_shape[i];
    linear_idx /= broadcasted_shape[j];
    if (strides1[j] != 0) idx1 += pos * strides1[j];
    if (strides2[j] != 0) idx2 += pos * strides2[j];
  }
  out[i] = a[idx1] * b[idx2];
}

__host__ void mul_broadcasted_tensor_cuda(Tensor* a, Tensor* b, float* out, int* broadcasted_shape, int broadcasted_size) {
  int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  int* strides1 = (int*)malloc(max_ndim * sizeof(int));
  int* strides2 = (int*)malloc(max_ndim * sizeof(int));
  if (strides1 == NULL || strides2 == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  int stride1 = 1, stride2 = 1;
  for (int i = max_ndim; i >=0 ; i--) {
    int dim1 = i<a->ndim ? a->shape[a->ndim - max_ndim + i] : 1;
    int dim2 = i<b->ndim ? b->shape[b->ndim - max_ndim + i] : 1;
    strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
    strides2[i] = dim1 == broadcasted_shape[i] ? stride2 : 0;
    stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
    stride2 *= (dim1 == broadcasted_shape[i]) ? dim2 : 1;
  }
  int *d_broadcasted_shape, *d_strides1, *d_stirdes2;

  cudaMalloc((void**)&d_broadcasted_shape, max_ndim * sizeof(int));
  cudaMemcpy(d_broadcasted_shape, broadcasted_shape, max_ndim * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_strides1, max_ndim * sizeof(int));
  cudaMemcpy(d_strides1, strides1, max_ndim * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_strides2, max_ndim * sizeof(int));
  cudaMemcpy(d_strides2, strides2, max_ndim * sizeof(int), cudaMemcpyHostToDevice);

  int n_of_blocks = (broadcasted_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  mul_broadcasted_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>>(a->a, b->a, out, d_broadcasted_shape, d_strides1, d_strides2, max_ndim, broadcasted_size);
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void div_broadcasted_tensor_cuda_kernel(float* a, float* b, float* out, int* broadcasted_shape, int* strides1, int* strides2, int max_dim, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= size) return;
  int idx1 = 0, idx2 = 0;
  int linear_idx = i;
  for (int j = max_dim - 1; j >= 0; j--) {
    int pos = linear_idx % brodacasted_shape[i];
    linear_idx /= broadcasted_shape[j];
    if (strides1[j] != 0) idx1 += pos * strides1[j];
    if (strides2[j] != 0) idx2 += pos * strides2[j];
  }
  out[i] = a[idx1] / b[idx2];
}

__host__ void div_broadcasted_tensor_cuda(Tensor* a, Tensor* b, float* out, int* broadcasted_shape, int broadcasted_size) {
  int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  int* strides1 = (int*)malloc(max_ndim * sizeof(int));
  int* strides2 = (int*)malloc(max_ndim * sizeof(int));
  if (strides1 == NULL || strides2 == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  int stride1 = 1, stride2 = 1;
  for (int i = max_ndim; i >=0 ; i--) {
    int dim1 = i<a->ndim ? a->shape[a->ndim - max_ndim + i] : 1;
    int dim2 = i<b->ndim ? b->shape[b->ndim - max_ndim + i] : 1;
    strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
    strides2[i] = dim1 == broadcasted_shape[i] ? stride2 : 0;
    stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
    stride2 *= (dim1 == broadcasted_shape[i]) ? dim2 : 1;
  }
  int *d_broadcasted_shape, *d_strides1, *d_stirdes2;

  cudaMalloc((void**)&d_broadcasted_shape, max_ndim * sizeof(int));
  cudaMemcpy(d_broadcasted_shape, broadcasted_shape, max_ndim * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_strides1, max_ndim * sizeof(int));
  cudaMemcpy(d_strides1, strides1, max_ndim * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_strides2, max_ndim * sizeof(int));
  cudaMemcpy(d_strides2, strides2, max_ndim * sizeof(int), cudaMemcpyHostToDevice);

  int n_of_blocks = (broadcasted_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  div_broadcasted_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>>(a->a, b->a, out, d_broadcasted_shape, d_strides1, d_strides2, max_ndim, broadcasted_size);
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void a_pow_scalar_cuda_kernel(float* a, float exp, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = powf(a[i], exp);
  }
}

__host__ void a_pow_scalar_cuda(Tensor* a, float exp, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  a_pow_scalar_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->a, exp, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void scalar_pow_tensor_cuda_kernel(float base, float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = powf(base, a[i]);
  }
}

__host__ void scalar_pow_tensor_cuda(float base, Tensor* a, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  scalar_pow_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(base, a->a, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void log_tensor_cuda_kernel(float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = logf(a[i]);
  }
}

__host__ void log_tensor_cuda(Tensor* a, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  log_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->a, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void sigmoid_tensor_cuda_kernel(float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = logf(a[i]);
  }
}

__host__ void sigmoid_tensor_cuda(Tensor* a, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  sigmoid_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->a, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void tanh_tensor_cuda_kernel(float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = tanh(a[i]);
  }
}

__host__ void tanh_tensor_cuda(Tensor* a, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  tanh_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->a, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void sin_tensor_cuda_kernel(float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = sinf(a[i]);
  }
}

__host__ void sin_tensor_cuda(Tensor* a, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  sin_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->a, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void cos_tensor_cuda_kernel(float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = cosf(a[i]);
  }
}

__host__ void cos_tensor_cuda(Tensor* a, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  cos_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->a, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void relu_tensor_cuda_kernel(float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = fmax(a[i], 0.0);
  }
}

__host__ void relu_tensor_cuda(Tensor* a, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  relu_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->a, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void zeros_like_tensor_cuda_kernel(float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = 0.0;
  }
}

__host__ void zeros_like_tensor_cuda(Tensor* a, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  zeros_like_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->a, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void ones_like_tensor_cuda_kernel(float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = 1.0;
  }
}

__host__ void ones_like_tensor_cuda(Tensor* a, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  ones_like_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->a, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void equal_tensor_cuda_kernel(float* a, float* b, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = (a[i] == b[i]) ? 1.0f : 0.0f;
  }
}

__host__ void equal_tensor_cuda(Tensor* a, Tensor* b float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  equal_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->a, b->a, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void equal_broadcasted_tensor_cuda_kernel(float* a, float* b, float* out, int* broadcasted_shape, int* strides1, int* strides2, int max_ndim, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= size) return;
  int idx1 = 0, idx2 = 0;
  int linear_idx = i;
  for (int j = max_dim - 1; j >= 0; j--) {
    int pos = linear_idx % brodacasted_shape[i];
    linear_idx /= broadcasted_shape[j];
    if (strides1[j] != 0) idx1 += pos * strides1[j];
    if (strides2[j] != 0) idx2 += pos * strides2[j];
  }
  out[i] = (a[idx1] == b[idx2]) ? 1.0f : 0.0f;
}

__host__ void equal_broadcasted_tensor_cuda(Tensor* a, Tensor* b, float* out, int* broadcasted_shape, int broadcasted_size) {
  int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  int* strides1 = (int*)malloc(max_ndim * sizeof(int));
  int* strides2 = (int*)malloc(max_ndim * sizeof(int));
  if (strides1 == NULL || strides2 == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  int stride1 = 1, stride2 = 1;
  for (int i = max_ndim; i >=0 ; i--) {
    int dim1 = i<a->ndim ? a->shape[a->ndim - max_ndim + i] : 1;
    int dim2 = i<b->ndim ? b->shape[b->ndim - max_ndim + i] : 1;
    strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
    strides2[i] = dim1 == broadcasted_shape[i] ? stride2 : 0;
    stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
    stride2 *= (dim1 == broadcasted_shape[i]) ? dim2 : 1;
  }
  int *d_broadcasted_shape, *d_strides1, *d_stirdes2;

  cudaMalloc((void**)&d_broadcasted_shape, max_ndim * sizeof(int));
  cudaMemcpy(d_broadcasted_shape, broadcasted_shape, max_ndim * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_strides1, max_ndim * sizeof(int));
  cudaMemcpy(d_strides1, strides1, max_ndim * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_strides2, max_ndim * sizeof(int));
  cudaMemcpy(d_strides2, strides2, max_ndim * sizeof(int), cudaMemcpyHostToDevice);

  int n_of_blocks = (broadcasted_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  equal_broadcasted_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>>(a->a, b->a, out, d_broadcasted_shape, d_strides1, d_strides2, max_ndim, broadcasted_size);
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void sum_tensor_cuda_kernel(float* a, float* out, int size) {
  __shared__ float partial_sum[THREADS_PER_BLOCK * sizeof(float)];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  partial_sum[tid] = (i < size) ? a[i] : 0;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) { // s>>=1 --> s = s/2
    if (tid < s) {
      partial_sum[tid] += partial_sum[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[blockIdx.x] = partial_sum[0];
  }
}

__global__ void sum_tensor_cuda_kernel_axis(float* a, float* out, int* strides, int* shape, int axis, int ndim, int axis_stride, int size, int result_size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < result_size) {
    for (int i = 0; i < shape[axis]; i++) {
      int index = 0;
      int remainder = tid;
      for (int k = ndim - 2; k >= 0; k--) {
        index += (remainder % shape[k < axis ? k : k + 1]) * strides[k < axis ? k : k + 1];
        remainder /= shape[k < axis ? k : k + 1];
      }
      index += i * axis_stride;
      atomicAdd(&out[tid], a[index]);
    }
  }
}

__host__ void sum_tensor_cuda(Tensor* a, float* out, int axis) {
  if (axis == -1) {
    cudaMemcpy(out, a->a, a->size * sizeof(float), cudaMemcpyHostToDevice);      
    int num_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // first-level reduction
    sum_tensor_cuda_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(a->a, out, a->size);
    // if necessary, perform multiple levels of reduction
    while (num_blocks > 1) {
      int num_blocks_next = (num_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      sum_tensor_cuda_kernel<<<num_blocks_next, THREADS_PER_BLOCK>>>(out, out, num_blocks);
      num_blocks = num_blocks_next;
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
    cudaDeviceSynchronize();
  } else {
    int axis_stride = a->strides[axis];
    // calculate the size of the resulting a
    int result_size = 1;
    for (int i = 0; i < a->ndim; i++) {
      if (i != axis) {
        result_size *= a->shape[i];
      }
    }
    // allocating memory for strides & shape on the device
    int *d_strides, *d_shape;
    cudaMalloc(&d_strides, a->ndim * sizeof(int));
    cudaMalloc(&d_shape, a->ndim * sizeof(int));
    cudaMemcpy(d_strides, a->strides, a->ndim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, a->shape, a->ndim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(out, 0, result_size * sizeof(float));
    int num_threads = result_size;
    int num_blocks = (num_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    sum_tensor_cuda_kernel_axis<<<num_blocks, THREADS_PER_BLOCK>>>(a->a, out, d_strides, d_shape, axis, a->ndim, axis_stride, a->size, result_size);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
    cudaDeviceSynchronize();
    cudaFree(d_strides);
    cudaFree(d_shape);
  }
}

__global__ void matmul_tensor_cuda_kernel(float* a, float* b, float* out, int rows1, int cols1, int cols2) {    
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

__host__ void matmul_tensor_cuda(Tensor* a, Tensor* b, float* out) {
  int rows1 = a->shape[0];
  int cols1 = a->shape[1];
  int cols2 = b->shape[1];

  dim3 threadsPerBlock(16, 16);
  dim3 n_of_blocks((cols2 + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows1 + threadsPerBlock.y - 1) / threadsPerBlock.y);
  matmul_tensor_cuda_kernel<<<n_of_blocks, threadsPerBlock>>>(a->a, b->a, out, rows1, cols1, cols2);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  cudaDeviceSynchronize();
}

__global__ void batched_matmul_tensor_cuda_kernel(float* a, float* b, float* out, int batch_size, int rows1, int cols1, int cols2) {
  int batch = blockIdx.z;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows1 && col < cols2) {
    float sum = 0.0f;
    for (int k = 0; k < cols1; ++k) {
      sum += a[batch * rows1 * cols1 + row * cols1 + k] * b[batch * cols1 * cols2 + k * cols2 + col];
    }
    out[batch * rows1 * cols2 + row * cols2 + col] = sum;
  }    
}

__host__ void batched_matmul_tensor_cuda(Tensor* a, Tensor* b, float* out) {
  int batch_size = b->shape[0];
  int rows1 = a->shape[1];
  int cols1 = a->shape[2];
  int cols2 = b->shape[2];

  dim3 threadsPerBlock(16, 16);
  dim3 n_of_blocks((cols2 + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows1 + threadsPerBlock.y - 1) / threadsPerBlock.y, batch_size);
  batched_matmul_tensor_cuda_kernel<<<n_of_blocks, threadsPerBlock>>>(a->a, b->a, out, batch_size, rows1, cols1, cols2);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  cudaDeviceSynchronize();
}

__global__ void broadcasted_batched_matmul_tensor_cuda_kernel(float* a, float* b, float* out, int batch_size, int rows1, int cols1, int cols2) {
  int batch = blockIdx.z;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows1 && col < cols2) {
    float sum = 0.0f;
    for (int k = 0; k < cols1; ++k) {
      sum += a[row * cols1 + k] * b[batch * cols1 * cols2 + k * cols2 + col];
    }
    out[batch * rows1 * cols2 + row * cols2 + col] = sum;
  }    
}

__host__ void broadcasted_batched_matmul_tensor_cuda(Tensor* a, Tensor* b, float* out) {
  int batch_size = b->shape[0];
  int rows1 = a->shape[0];
  int cols1 = a->shape[1];
  int cols2 = b->shape[2];

  dim3 threadsPerBlock(16, 16);
  dim3 n_of_blocks((cols2 + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows1 + threadsPerBlock.y - 1) / threadsPerBlock.y, batch_size);
  broadcasted_batched_matmul_tensor_cuda_kernel<<<n_of_blocks, threadsPerBlock>>>(a->a, b->a, out, batch_size, rows1, cols1, cols2);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  cudaDeviceSynchronize();
}

__global__ void transpose_1D_tensor_cuda_kernel(float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = a[i];
  }
}

__host__ void transpose_1D_tensor_cuda(Tensor* a, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  transpose_1D_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->a, out, a->size);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  cudaDeviceSynchronize();
}

__global__ void transpose_2D_tensor_cuda_kernel(float* a, float* out, int rows, int cols) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < rows && j < cols) {
    out[j * rows + i] = a[i * cols + j];
  }
}

__host__ void transpose_2D_tensor_cuda(Tensor* a, float* out) {
  int rows = a->shape[0];
  int cols = a->shape[1];

  dim3 threadsPerBlock(16, 16);
  dim3 n_of_blocks((rows + threadsPerBlock.x - 1) / threadsPerBlock.x, (cols + threadsPerBlock.y - 1) / threadsPerBlock.y);
  transpose_2D_tensor_cuda_kernel<<<n_of_blocks, threadsPerBlock>>>(a->a, out, rows, cols);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  cudaDeviceSynchronize();
}

__global__ void transpose_3D_tensor_cuda_kernel(float* a, float* out, int batch, int rows, int cols) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < batch && j < rows && k < cols) {
    out[k * rows * batch + j * batch + i] = a[i * rows * cols + j * cols + k];
  }
}

__host__ void transpose_3D_tensor_cuda(Tensor* a, float* out) {
  int batch = a->shape[0];
  int rows = a->shape[1];
  int cols = a->shape[2];

  dim3 threadsPerBlock(8, 8, 8);
  dim3 n_of_blocks((batch + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y, (cols + threadsPerBlock.z - 1) / threadsPerBlock.z);
  transpose_3D_tensor_cuda_kernel<<<n_of_blocks, threadsPerBlock>>>(a->a, out, batch, rows, cols);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}


__global__ void assign_tensor_cuda_kernel(float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = a[i];
  }
}

__host__ void assign_tensor_cuda(Tensor* a, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  assign_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->a, out, a->size);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void make_contiguous_tensor_cuda_kernel(float* a, float* out, int ndim, int size, int* strides, int* new_strides) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    int index = 0;
    int offset = i;
    for (int j = 0; j < ndim; j++) {
      index += (offset / new_strides[j]) * strides[j];
      offset %= new_strides[j];
    }
    out[i] = a[index];
  }
}

__host__ void make_contiguous_tensor_cuda(Tensor* a, float* out, int* new_strides) {
  int* d_strides;
  cudaMalloc((void **)&d_strides, a->ndim * sizeof(int));
  cudaMemcpy(d_strides, a->strides, a->ndim * sizeof(int), cudaMemcpyHostToDevice);
  
  int* d_new_strides;
  cudaMalloc((void **)&d_new_strides, a->ndim * sizeof(int));
  cudaMemcpy(d_new_strides, new_strides, a->ndim * sizeof(int), cudaMemcpyHostToDevice);

  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  make_contiguous_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->a, out, a->ndim, a->size, d_strides, d_new_strides);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  cudaDeviceSynchronize();
  cudaFree(a->a);
  free(a->strides);
  a->a = out;
  a->strides = new_strides;
}