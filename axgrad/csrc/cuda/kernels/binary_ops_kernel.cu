#include <math.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include "binary_ops_kernel.cuh"
#include "../cuda.cuh"

__global__ void __add_kernel__(float* a, float* b, float* out, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = a[idx] + b[idx];
  }
}

__host__ void add_tensor_cuda(float* a, float* b, float* out, size_t size) {
  int threadsPerBlock = 256;
  int n_of_blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
  __add_kernel__<<<n_of_blocks, threadsPerBlock>>>(a, b, out, size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __add_scalar_kernel__(float* a, float b, float* out, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = a[idx] + b;
  }
}

__host__ void add_scalar_tensor_cuda(float* a, float b, float* out, size_t size) {
  int threadsPerBlock = 256;
  int n_of_blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
  __add_scalar_kernel__<<<n_of_blocks, threadsPerBlock>>>(a, b, out, size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __sub_kernel__(float* a, float* b, float* out, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = a[idx] - b[idx];
  }
}

__host__ void sub_tensor_cuda(float* a, float* b, float* out, size_t size) {
  int threadsPerBlock = 256;
  int n_of_blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
  __sub_kernel__<<<n_of_blocks, threadsPerBlock>>>(a, b, out, size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __sub_scalar_kernel__(float* a, float b, float* out, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = a[idx] - b;
  }
}

__host__ void sub_scalar_tensor_cuda(float* a, float b, float* out, size_t size) {
  int threadsPerBlock = 256;
  int n_of_blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
  __sub_scalar_kernel__<<<n_of_blocks, threadsPerBlock>>>(a, b, out, size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __mul_kernel__(float* a, float* b, float* out, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = a[idx] * b[idx];
  }
}

__host__ void mul_tensor_cuda(float* a, float* b, float* out, size_t size) {
  int threadsPerBlock = 256;
  int n_of_blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
  __mul_kernel__<<<n_of_blocks, threadsPerBlock>>>(a, b, out, size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __mul_scalar_kernel__(float* a, float b, float* out, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = a[idx] * b;
  }
}

__host__ void mul_scalar_tensor_cuda(float* a, float b, float* out, size_t size) {
  int threadsPerBlock = 256;
  int n_of_blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
  __mul_scalar_kernel__<<<n_of_blocks, threadsPerBlock>>>(a, b, out, size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __div_kernel__(float* a, float* b, float* out, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    if (b[idx] == 0.0f) {
      if (a[idx] > 0.0f) {
        out[idx] = INFINITY;
      } else if (a[idx] < 0.0f) {
        out[idx] = -INFINITY;
      } else {
        out[idx] = NAN;  // 0/0 case
      }
    } else {
      out[idx] = a[idx] / b[idx];
    }
  }
}

__host__ void div_tensor_cuda(float* a, float* b, float* out, size_t size) {
  int threadsPerBlock = 256;
  int n_of_blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
  __div_kernel__<<<n_of_blocks, threadsPerBlock>>>(a, b, out, size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __div_scalar_kernel__(float* a, float b, float* out, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    if (b == 0.0f) {
      if (a[idx] > 0.0f) {
        out[idx] = INFINITY;
      } else if (a[idx] < 0.0f) {
        out[idx] = -INFINITY;
      } else {
        out[idx] = NAN;  // 0/0 case
      }
    } else {
      out[idx] = a[idx] / b;
    }
  }
}

__host__ void div_scalar_tensor_cuda(float* a, float b, float* out, size_t size) {
  int threadsPerBlock = 256;
  int n_of_blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
  __div_scalar_kernel__<<<n_of_blocks, threadsPerBlock>>>(a, b, out, size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __pow_tensor_scalar_kernel__(float* a, float exp, float* out, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = powf(a[idx], exp);
  }
}

__host__ void pow_tensor_scalar_cuda(float* a, float exp, float* out, size_t size) {
  int threadsPerBlock = 256;
  int n_of_blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
  __pow_tensor_scalar_kernel__<<<n_of_blocks, threadsPerBlock>>>(a, exp, out, size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __pow_scalar_tensor_kernel__(float base, float* exp, float* out, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = powf(base, exp[idx]);
  }
}

__host__ void pow_scalar_tensor_cuda(float base, float* exp, float* out, size_t size) {
  int threadsPerBlock = 256;
  int n_of_blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
  __pow_scalar_tensor_kernel__<<<n_of_blocks, threadsPerBlock>>>(base, exp, out, size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaDeviceSynchronize();
}

__global__ void __add_broadcasted_kernel__(float* a, float* b, float* out, int* a_strides, int* b_strides, int* broadcasted_shape, int ndim, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int linear_index = idx;
    int a_index = 0, b_index = 0;
    
    for (int j = ndim - 1; j >= 0; j--) {
      int pos = linear_index % broadcasted_shape[j];
      linear_index /= broadcasted_shape[j];
      a_index += pos * a_strides[j];
      b_index += pos * b_strides[j];
    }
    
    out[idx] = a[a_index] + b[b_index];
  }
}

__host__ void add_broadcasted_tensor_cuda(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape) {
  int max_ndim = a_ndim > b_ndim ? a_ndim : b_ndim;
  
  int* h_a_strides = (int*)malloc(max_ndim * sizeof(int));
  int* h_b_strides = (int*)malloc(max_ndim * sizeof(int));
  int* d_a_strides;
  int* d_b_strides;
  int* d_broadcasted_shape;
  
  if (h_a_strides == NULL || h_b_strides == NULL) {
    fprintf(stderr, "Couldn't assign the strides to memory, operation failed!\n");
    exit(-1);
  }
  
  int stride_a = 1, stride_b = 1;
  for (int i = max_ndim - 1; i >= 0; i--) {
    int dim_a = (a_ndim - max_ndim + i >= 0) ? a_shape[a_ndim - max_ndim + i] : 1;
    int dim_b = (b_ndim - max_ndim + i >= 0) ? b_shape[b_ndim - max_ndim + i] : 1;
    
    h_a_strides[i] = (dim_a == broadcasted_shape[i]) ? stride_a : 0;
    h_b_strides[i] = (dim_b == broadcasted_shape[i]) ? stride_b : 0;
    
    stride_a *= (dim_a == broadcasted_shape[i]) ? dim_a : 1;
    stride_b *= (dim_b == broadcasted_shape[i]) ? dim_b : 1;
  }
  
  cudaMalloc(&d_a_strides, max_ndim * sizeof(int));
  cudaMalloc(&d_b_strides, max_ndim * sizeof(int));
  cudaMalloc(&d_broadcasted_shape, max_ndim * sizeof(int));
  
  cudaMemcpy(d_a_strides, h_a_strides, max_ndim * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_strides, h_b_strides, max_ndim * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_broadcasted_shape, broadcasted_shape, max_ndim * sizeof(int), cudaMemcpyHostToDevice);
  
  int threadsPerBlock = 256;
  int n_of_blocks = (broadcasted_size + threadsPerBlock - 1) / threadsPerBlock;
  __add_broadcasted_kernel__<<<n_of_blocks, threadsPerBlock>>>(a, b, out, d_a_strides, d_b_strides, d_broadcasted_shape, max_ndim, broadcasted_size);
  
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  
  cudaDeviceSynchronize();
  
  free(h_a_strides);
  free(h_b_strides);
  cudaFree(d_a_strides);
  cudaFree(d_b_strides);
  cudaFree(d_broadcasted_shape);
}

__global__ void __sub_broadcasted_kernel__(float* a, float* b, float* out, int* a_strides, int* b_strides, int* broadcasted_shape, int ndim, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int linear_index = idx;
    int a_index = 0, b_index = 0;
    
    for (int j = ndim - 1; j >= 0; j--) {
      int pos = linear_index % broadcasted_shape[j];
      linear_index /= broadcasted_shape[j];
      a_index += pos * a_strides[j];
      b_index += pos * b_strides[j];
    }
    
    out[idx] = a[a_index] - b[b_index];
  }
}

__host__ void sub_broadcasted_tensor_cuda(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape) {
  int max_ndim = a_ndim > b_ndim ? a_ndim : b_ndim;
  
  int* h_a_strides = (int*)malloc(max_ndim * sizeof(int));
  int* h_b_strides = (int*)malloc(max_ndim * sizeof(int));
  int* d_a_strides;
  int* d_b_strides;
  int* d_broadcasted_shape;
  
  if (h_a_strides == NULL || h_b_strides == NULL) {
    fprintf(stderr, "Couldn't assign the strides to memory, operation failed!\n");
    exit(-1);
  }
  
  int stride_a = 1, stride_b = 1;
  for (int i = max_ndim - 1; i >= 0; i--) {
    int dim_a = (a_ndim - max_ndim + i >= 0) ? a_shape[a_ndim - max_ndim + i] : 1;
    int dim_b = (b_ndim - max_ndim + i >= 0) ? b_shape[b_ndim - max_ndim + i] : 1;
    
    h_a_strides[i] = (dim_a == broadcasted_shape[i]) ? stride_a : 0;
    h_b_strides[i] = (dim_b == broadcasted_shape[i]) ? stride_b : 0;
    
    stride_a *= (dim_a == broadcasted_shape[i]) ? dim_a : 1;
    stride_b *= (dim_b == broadcasted_shape[i]) ? dim_b : 1;
  }
  
  cudaMalloc(&d_a_strides, max_ndim * sizeof(int));
  cudaMalloc(&d_b_strides, max_ndim * sizeof(int));
  cudaMalloc(&d_broadcasted_shape, max_ndim * sizeof(int));
  
  cudaMemcpy(d_a_strides, h_a_strides, max_ndim * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_strides, h_b_strides, max_ndim * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_broadcasted_shape, broadcasted_shape, max_ndim * sizeof(int), cudaMemcpyHostToDevice);
  
  int threadsPerBlock = 256;
  int n_of_blocks = (broadcasted_size + threadsPerBlock - 1) / threadsPerBlock;
  __sub_broadcasted_kernel__<<<n_of_blocks, threadsPerBlock>>>(a, b, out, d_a_strides, d_b_strides, d_broadcasted_shape, max_ndim, broadcasted_size);
  
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  
  cudaDeviceSynchronize();
  
  free(h_a_strides);
  free(h_b_strides);
  cudaFree(d_a_strides);
  cudaFree(d_b_strides);
  cudaFree(d_broadcasted_shape);
}

__global__ void __mul_broadcasted_kernel__(float* a, float* b, float* out, int* a_strides, int* b_strides, int* broadcasted_shape, int ndim, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int linear_index = idx;
    int a_index = 0, b_index = 0;
    
    for (int j = ndim - 1; j >= 0; j--) {
      int pos = linear_index % broadcasted_shape[j];
      linear_index /= broadcasted_shape[j];
      a_index += pos * a_strides[j];
      b_index += pos * b_strides[j];
    }
    
    out[idx] = a[a_index] * b[b_index];
  }
}

__host__ void mul_broadcasted_tensor_cuda(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape) {
  int max_ndim = a_ndim > b_ndim ? a_ndim : b_ndim;
  
  int* h_a_strides = (int*)malloc(max_ndim * sizeof(int));
  int* h_b_strides = (int*)malloc(max_ndim * sizeof(int));
  int* d_a_strides;
  int* d_b_strides;
  int* d_broadcasted_shape;
  
  if (h_a_strides == NULL || h_b_strides == NULL) {
    fprintf(stderr, "Couldn't assign the strides to memory, operation failed!\n");
    exit(-1);
  }
  
  int stride_a = 1, stride_b = 1;
  for (int i = max_ndim - 1; i >= 0; i--) {
    int dim_a = (a_ndim - max_ndim + i >= 0) ? a_shape[a_ndim - max_ndim + i] : 1;
    int dim_b = (b_ndim - max_ndim + i >= 0) ? b_shape[b_ndim - max_ndim + i] : 1;
    
    h_a_strides[i] = (dim_a == broadcasted_shape[i]) ? stride_a : 0;
    h_b_strides[i] = (dim_b == broadcasted_shape[i]) ? stride_b : 0;
    
    stride_a *= (dim_a == broadcasted_shape[i]) ? dim_a : 1;
    stride_b *= (dim_b == broadcasted_shape[i]) ? dim_b : 1;
  }
  
  cudaMalloc(&d_a_strides, max_ndim * sizeof(int));
  cudaMalloc(&d_b_strides, max_ndim * sizeof(int));
  cudaMalloc(&d_broadcasted_shape, max_ndim * sizeof(int));
  
  cudaMemcpy(d_a_strides, h_a_strides, max_ndim * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_strides, h_b_strides, max_ndim * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_broadcasted_shape, broadcasted_shape, max_ndim * sizeof(int), cudaMemcpyHostToDevice);
  
  int threadsPerBlock = 256;
  int n_of_blocks = (broadcasted_size + threadsPerBlock - 1) / threadsPerBlock;
  __mul_broadcasted_kernel__<<<n_of_blocks, threadsPerBlock>>>(a, b, out, d_a_strides, d_b_strides, d_broadcasted_shape, max_ndim, broadcasted_size);
  
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  
  cudaDeviceSynchronize();
  
  free(h_a_strides);
  free(h_b_strides);
  cudaFree(d_a_strides);
  cudaFree(d_b_strides);
  cudaFree(d_broadcasted_shape);
}

__global__ void __div_broadcasted_kernel__(float* a, float* b, float* out, int* a_strides, int* b_strides, int* broadcasted_shape, int ndim, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int linear_index = idx;
    int a_index = 0, b_index = 0;
    
    for (int j = ndim - 1; j >= 0; j--) {
      int pos = linear_index % broadcasted_shape[j];
      linear_index /= broadcasted_shape[j];
      a_index += pos * a_strides[j];
      b_index += pos * b_strides[j];
    }
    
    if (b[b_index] == 0.0f) {
      if (a[a_index] > 0.0f) {
        out[idx] = INFINITY;
      } else if (a[a_index] < 0.0f) {
        out[idx] = -INFINITY;
      } else {
        out[idx] = NAN;  // 0/0 case
      }
    } else {
      out[idx] = a[a_index] / b[b_index];
    }
  }
}

__host__ void div_broadcasted_tensor_cuda(float* a, float* b, float* out, int* broadcasted_shape, int broadcasted_size, int a_ndim, int b_ndim, int* a_shape, int* b_shape) {
  int max_ndim = a_ndim > b_ndim ? a_ndim : b_ndim;
  
  int* h_a_strides = (int*)malloc(max_ndim * sizeof(int));
  int* h_b_strides = (int*)malloc(max_ndim * sizeof(int));
  int* d_a_strides;
  int* d_b_strides;
  int* d_broadcasted_shape;
  
  if (h_a_strides == NULL || h_b_strides == NULL) {
    fprintf(stderr, "Couldn't assign the strides to memory, operation failed!\n");
    exit(-1);
  }
  
  int stride_a = 1, stride_b = 1;
  for (int i = max_ndim - 1; i >= 0; i--) {
    int dim_a = (a_ndim - max_ndim + i >= 0) ? a_shape[a_ndim - max_ndim + i] : 1;
    int dim_b = (b_ndim - max_ndim + i >= 0) ? b_shape[b_ndim - max_ndim + i] : 1;

    h_a_strides[i] = (dim_a == broadcasted_shape[i]) ? stride_a : 0;
    h_b_strides[i] = (dim_b == broadcasted_shape[i]) ? stride_b : 0;
    
    stride_a *= (dim_a == broadcasted_shape[i]) ? dim_a : 1;
    stride_b *= (dim_b == broadcasted_shape[i]) ? dim_b : 1;
  }
  
  cudaMalloc(&d_a_strides, max_ndim * sizeof(int));
  cudaMalloc(&d_b_strides, max_ndim * sizeof(int));
  cudaMalloc(&d_broadcasted_shape, max_ndim * sizeof(int));
  
  cudaMemcpy(d_a_strides, h_a_strides, max_ndim * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_strides, h_b_strides, max_ndim * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_broadcasted_shape, broadcasted_shape, max_ndim * sizeof(int), cudaMemcpyHostToDevice);
  
  int threadsPerBlock = 256;
  int n_of_blocks = (broadcasted_size + threadsPerBlock - 1) / threadsPerBlock;
  __div_broadcasted_kernel__<<<n_of_blocks, threadsPerBlock>>>(a, b, out, d_a_strides, d_b_strides, d_broadcasted_shape, max_ndim, broadcasted_size);
  
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  cudaDeviceSynchronize();

  free(h_a_strides);
  free(h_b_strides);
  cudaFree(d_a_strides);
  cudaFree(d_b_strides);
  cudaFree(d_broadcasted_shape);
}