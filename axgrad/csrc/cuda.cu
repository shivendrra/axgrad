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

__global__ void scalar_div_tensor_cuda_kernel(float scalar, float* a, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = scalar / a[i];
  }
}

__host__ void scalar_div_tensor_cuda(float scalar, Tensor* b, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  scalar_div_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(scalar, a->data, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void tensor_div_scalar_cuda_kernel(float* a, float scalar, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = a[i] / scalar;
  }
}

__host__ void tensor_div_scalar_cuda(float scalar, Tensor* b, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  tensor_div_scalar_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->data, scalar, out, a->size);
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
  scalar_mul_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCKS>>>(scalar, a->data, out, a->size);
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
  add_broadcasted_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>>(a->data, b->data, out, d_broadcasted_shape, d_strides1, d_strides2, max_ndim, broadcasted_size);
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
  sub_broadcasted_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>>(a->data, b->data, out, d_broadcasted_shape, d_strides1, d_strides2, max_ndim, broadcasted_size);
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
  mul_broadcasted_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>>(a->data, b->data, out, d_broadcasted_shape, d_strides1, d_strides2, max_ndim, broadcasted_size);
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void tensor_pow_scalar_cuda_kernel(float* a, float exp, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = powf(data[i], exp);
  }
}

__host__ void tensor_pow_scalar_cuda(Tensor* a, float exp, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  tensor_pow_scalar_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->data, exp, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void scalar_pow_tensor_cuda_kernel(float base, float* data, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = powf(base, data[i]);
  }
}

__host__ void scalar_pow_tensor_cuda(float base, Tensor* a, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  scalar_pow_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(base, a->data, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void log_tensor_cuda_kernel(float* data, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = logf(data[i]);
  }
}

__host__ void log_tensor_cuda(Tensor* a, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  log_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->data, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void sigmoid_tensor_cuda_kernel(float* data, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = logf(data[i]);
  }
}

__host__ void sigmoid_tensor_cuda(Tensor* a, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  sigmoid_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->data, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void tanh_tensor_cuda_kernel(float* data, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = tanh(data[i]);
  }
}

__host__ void tanh_tensor_cuda(Tensor* a, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  tanh_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->data, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void sin_tensor_cuda_kernel(float* data, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = sinf(data[i]);
  }
}

__host__ void sin_tensor_cuda(Tensor* a, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  sin_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->data, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void cos_tensor_cuda_kernel(float* data, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = cosf(data[i]);
  }
}

__host__ void cos_tensor_cuda(Tensor* a, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  cos_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->data, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void relu_tensor_cuda_kernel(float* data, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = fmax(data[i], 0.0);
  }
}

__host__ void relu_tensor_cuda(Tensor* a, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  relu_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->data, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void zeros_like_tensor_cuda_kernel(float* data, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = 0.0;
  }
}

__host__ void zeros_like_tensor_cuda(Tensor* a, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  zeros_like_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->data, out, a->size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void ones_like_tensor_cuda_kernel(float* data, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = 1.0;
  }
}

__host__ void ones_like_tensor_cuda(Tensor* a, float* out) {
  int n_of_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  ones_like_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->data, out, a->size);
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
  equal_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>(a->data, b->data, out, a->size);
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
  equal_broadcasted_tensor_cuda_kernel<<<n_of_blocks, THREADS_PER_BLOCK>>>>(a->data, b->data, out, d_broadcasted_shape, d_strides1, d_strides2, max_ndim, broadcasted_size);
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}