#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cfloat>
#include "red_ops_kernel.cuh"
#include "../cuda.cuh"

__global__ void __sum_kernel__(float* a, float* out, int size) {
  __shared__ float partial_sum[THREADS_PER_BLOCK];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  partial_sum[tid] = (i < size) ? a[i] : 0.0f;
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

__global__ void __sum_axis_kernel__(float* a, float* out, int* strides, int* shape, int axis, int ndim, int axis_stride, int size, int result_size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < result_size) {
    for (int i = 0; i < shape[axis]; i++) {
      int index = 0;
      int remainder = tid;
      for (int k = ndim - 2; k >= 0; k--) {
        int dim_idx = (k < axis) ? k : k + 1;
        index += (remainder % shape[dim_idx]) * strides[dim_idx];
        remainder /= shape[dim_idx];
      }
      index += i * axis_stride;
      atomicAdd(&out[tid], a[index]);
    }
  }
}

__host__ void sum_tensor_cuda(float* a, float* out, int axis, size_t size, int* strides, int* shape, size_t ndim) {
  if (axis == -1) {
    // Global reduction
    int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    float* temp_out;
    cudaMalloc(&temp_out, num_blocks * sizeof(float));

    // First-level reduction
    __sum_kernel__<<<num_blocks, THREADS_PER_BLOCK>>>(a, temp_out, size);

    // continuing reduction until single value
    while (num_blocks > 1) {
      int num_blocks_next = (num_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      float* next_temp;
      cudaMalloc(&next_temp, num_blocks_next * sizeof(float));
      __sum_kernel__<<<num_blocks_next, THREADS_PER_BLOCK>>>(temp_out, next_temp, num_blocks);
      cudaFree(temp_out);
      temp_out = next_temp;
      num_blocks = num_blocks_next;
    }

    cudaMemcpy(out, temp_out, sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(temp_out);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
    cudaDeviceSynchronize();
  } else {
    // Axis-specific reduction
    int axis_stride = strides[axis];

    // calculating the size of the resulting array
    int result_size = 1;
    for (int i = 0; i < ndim; i++) {
      if (i != axis) {
        result_size *= shape[i];
      }
    }

    // allocating memory for strides & shape on the device
    int *d_strides, *d_shape;
    cudaMalloc(&d_strides, ndim * sizeof(int));
    cudaMalloc(&d_shape, ndim * sizeof(int));
    cudaMemcpy(d_strides, strides, ndim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, shape, ndim * sizeof(int), cudaMemcpyHostToDevice);

    // initializing output array to zero
    cudaMemset(out, 0, result_size * sizeof(float));

    int num_blocks = (result_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    __sum_axis_kernel__<<<num_blocks, THREADS_PER_BLOCK>>>(a, out, d_strides, d_shape, axis, ndim, axis_stride, size, result_size);

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

__global__ void __max_kernel__(float* a, float* out, int size) {
  __shared__ float partial_max[THREADS_PER_BLOCK];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  partial_max[tid] = (i < size) ? a[i] : -FLT_MAX;
  __syncthreads();
  
  // block-wise reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      partial_max[tid] = fmaxf(partial_max[tid], partial_max[tid + s]);
    }
    __syncthreads();
  }

  // block result to global memory
  if (tid == 0) {
    out[blockIdx.x] = partial_max[0];
  }
}

__device__ float __atomicMaxFloat__(float* address, float val) {
  int* address_as_int = (int*)address;
  int old = *address_as_int, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__global__ void __max_axis_kernel__(float* a, float* out, int* strides, int* shape, int axis, int ndim, int axis_stride, int size, int result_size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < result_size) {
    for (int i = 0; i < shape[axis]; i++) {
      int index = 0;
      int remainder = tid;
      for (int k = ndim - 2; k >= 0; k--) {
        int dim_idx = (k < axis) ? k : k + 1;
        index += (remainder % shape[dim_idx]) * strides[dim_idx];
        remainder /= shape[dim_idx];
      }
      index += i * axis_stride;
      __atomicMaxFloat__(&out[tid], a[index]);
    }
  }
}

__host__ void max_tensor_cuda(float* a, float* out, int axis, size_t size, int* strides, int* shape, size_t ndim) {
  if (axis == -1) {
    // Global reduction
    int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    float* temp_out;
    cudaMalloc(&temp_out, num_blocks * sizeof(float));

    // First-level reduction
    __max_kernel__<<<num_blocks, THREADS_PER_BLOCK>>>(a, temp_out, size);

    // continuing reduction until single value
    while (num_blocks > 1) {
      int num_blocks_next = (num_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      float* next_temp;
      cudaMalloc(&next_temp, num_blocks_next * sizeof(float));
      __max_kernel__<<<num_blocks_next, THREADS_PER_BLOCK>>>(temp_out, next_temp, num_blocks);
      cudaFree(temp_out);
      temp_out = next_temp;
      num_blocks = num_blocks_next;
    }

    cudaMemcpy(out, temp_out, sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(temp_out);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
    cudaDeviceSynchronize();      
  } else {
    // Axis-specific reduction
    int axis_stride = strides[axis];
    int result_size = 1;
    for (int i = 0; i < ndim; i++) {
      if (i != axis) {
        result_size *= shape[i];
      }
    }

    int *d_strides, *d_shape;
    cudaMalloc(&d_strides, ndim * sizeof(int));
    cudaMalloc(&d_shape, ndim * sizeof(int));
    cudaMemcpy(d_strides, strides, ndim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, shape, ndim * sizeof(int), cudaMemcpyHostToDevice);

    // initializing output with -FLT_MAX
    float neg_inf = -FLT_MAX;
    int* neg_inf_as_int = reinterpret_cast<int*>(&neg_inf);
    for (int i = 0; i < result_size; i++) {
      cudaMemcpy(out + i, neg_inf_as_int, sizeof(float), cudaMemcpyHostToDevice);
    }

    int num_blocks = (result_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    __max_axis_kernel__<<<num_blocks, THREADS_PER_BLOCK>>>(a, out, d_strides, d_shape, axis, ndim, axis_stride, size, result_size);

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

__global__ void __min_kernel__(float* a, float* out, int size) {
  __shared__ float partial_min[THREADS_PER_BLOCK];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  partial_min[tid] = (i < size) ? a[i] : FLT_MAX;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      partial_min[tid] = fminf(partial_min[tid], partial_min[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[blockIdx.x] = partial_min[0];
  }
}

__device__ float __atomicMinFloat__(float* address, float val) {
  int* address_as_int = (int*)address;
  int old = *address_as_int, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__global__ void __min_axis_kernel__(float* a, float* out, int* strides, int* shape, int axis, int ndim, int axis_stride, int size, int result_size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < result_size) {
    for (int i = 0; i < shape[axis]; i++) {
      int index = 0;
      int remainder = tid;
      for (int k = ndim - 2; k >= 0; k--) {
        int dim_idx = (k < axis) ? k : k + 1;
        index += (remainder % shape[dim_idx]) * strides[dim_idx];
        remainder /= shape[dim_idx];
      }
      index += i * axis_stride;
      __atomicMinFloat__(&out[tid], a[index]);
    }
  }
}

__host__ void min_tensor_cuda(float* a, float* out, int axis, size_t size, int* strides, int* shape, size_t ndim) {
  if (axis == -1) {
    // Global reduction
    int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    float* temp_out;
    cudaMalloc(&temp_out, num_blocks * sizeof(float));

    // First-level reduction
    __min_kernel__<<<num_blocks, THREADS_PER_BLOCK>>>(a, temp_out, size);

    // continuing reduction until single value
    while (num_blocks > 1) {
      int num_blocks_next = (num_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      float* next_temp;
      cudaMalloc(&next_temp, num_blocks_next * sizeof(float));
      __min_kernel__<<<num_blocks_next, THREADS_PER_BLOCK>>>(temp_out, next_temp, num_blocks);
      cudaFree(temp_out);
      temp_out = next_temp;
      num_blocks = num_blocks_next;
    }
    
    cudaMemcpy(out, temp_out, sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(temp_out);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
    cudaDeviceSynchronize();  
  } else {
    // Axis-specific reduction
    int axis_stride = strides[axis];
    int result_size = 1;
    for (int i = 0; i < ndim; i++) {
      if (i != axis) {
        result_size *= shape[i];
      }
    }

    int *d_strides, *d_shape;
    cudaMalloc(&d_strides, ndim * sizeof(int));
    cudaMalloc(&d_shape, ndim * sizeof(int));
    cudaMemcpy(d_strides, strides, ndim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, shape, ndim * sizeof(int), cudaMemcpyHostToDevice);

    // initializing output with FLT_MAX
    float inf = FLT_MAX;
    int* inf_as_int = reinterpret_cast<int*>(&inf);
    for (int i = 0; i < result_size; i++) {
      cudaMemcpy(out + i, inf_as_int, sizeof(float), cudaMemcpyHostToDevice);
    }

    int num_blocks = (result_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    __min_axis_kernel__<<<num_blocks, THREADS_PER_BLOCK>>>(a, out, d_strides, d_shape, axis, ndim, axis_stride, size, result_size);

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