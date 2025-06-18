#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include "stats_ops_kernel.cuh"
#include "../cuda.cuh"

// Helper kernel for element-wise division
__global__ void __divide_kernel__(float* arr, int size, float divisor) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    arr[tid] /= divisor;
  }
}

// ====================== MEAN OPS ======================

__global__ void __mean_kernel__(float* a, float* out, int size) {
  __shared__ float partial_sum[THREADS_PER_BLOCK];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  partial_sum[tid] = (i < size) ? a[i] : 0.0f;
  __syncthreads();

  // Block-wise reduction (sum)
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      partial_sum[tid] += partial_sum[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[blockIdx.x] = partial_sum[0];
  }
}

__global__ void __mean_axis_kernel__(float* a, float* out, int* strides, int* shape, int axis, int ndim, int axis_stride, int size, int result_size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < result_size) {
    float sum = 0.0f;
    for (int i = 0; i < shape[axis]; i++) {
      int index = 0;
      int remainder = tid;
      for (int k = ndim - 2; k >= 0; k--) {
        int dim_idx = (k < axis) ? k : k + 1;
        index += (remainder % shape[dim_idx]) * strides[dim_idx];
        remainder /= shape[dim_idx];
      }
      index += i * axis_stride;
      sum += a[index];
    }
    out[tid] = sum / shape[axis];  // Divide by axis size to get mean
  }
}

__host__ void mean_tensor_cuda(float* a, float* out, int axis, size_t size, int* strides, int* shape, size_t ndim) {
  if (axis == -1) {
    // Global mean
    int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    float* temp_out;
    cudaMalloc(&temp_out, num_blocks * sizeof(float));
    
    // First-level reduction (sum)
    __mean_kernel__<<<num_blocks, THREADS_PER_BLOCK>>>(a, temp_out, size);
    
    // continuing reduction until single value
    while (num_blocks > 1) {
      int num_blocks_next = (num_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      float* next_temp;
      cudaMalloc(&next_temp, num_blocks_next * sizeof(float));
      __mean_kernel__<<<num_blocks_next, THREADS_PER_BLOCK>>>(temp_out, next_temp, num_blocks);
      cudaFree(temp_out);
      temp_out = next_temp;
      num_blocks = num_blocks_next;
    }

    // copying result and divide by total size to get mean
    float sum;
    cudaMemcpy(&sum, temp_out, sizeof(float), cudaMemcpyDeviceToHost);
    float mean = sum / size;
    cudaMemcpy(out, &mean, sizeof(float), cudaMemcpyHostToDevice);
    cudaFree(temp_out);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
    cudaDeviceSynchronize();
  } else {
    // Axis-specific mean
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
    
    int num_blocks = (result_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    __mean_axis_kernel__<<<num_blocks, THREADS_PER_BLOCK>>>(a, out, d_strides, d_shape, axis, ndim, axis_stride, size, result_size);
    
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

// ====================== VAR OPS ======================

__global__ void __var_mean_kernel__(float* a, float* means, int size) {
  __shared__ float partial_sum[THREADS_PER_BLOCK];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  partial_sum[tid] = (i < size) ? a[i] : 0.0f;
  __syncthreads();
  
  // Block-wise reduction (sum for mean calculation)
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      partial_sum[tid] += partial_sum[tid + s];
    }
    __syncthreads();  
  }
  
  if (tid == 0) {
    means[blockIdx.x] = partial_sum[0];
  }
}

__global__ void __var_kernel__(float* a, float* means, float* out, int size) {
  __shared__ float partial_var[THREADS_PER_BLOCK];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i < size) {
    float diff = a[i] - means[0];  // means[0] contains the global mean
    partial_var[tid] = diff * diff;
  } else {
    partial_var[tid] = 0.0f;
  }
  __syncthreads();
  
  // Block-wise reduction (sum of squared differences)
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      partial_var[tid] += partial_var[tid + s];
    }
    __syncthreads();
  }
  
  if (tid == 0) {
    out[blockIdx.x] = partial_var[0];
  }
}

__global__ void __var_axis_mean_kernel__(float* a, float* means, int* strides, int* shape, int axis, int ndim, int axis_stride, int size, int result_size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < result_size) {
    float sum = 0.0f;
    for (int i = 0; i < shape[axis]; i++) {
      int index = 0;
      int remainder = tid;
      for (int k = ndim - 2; k >= 0; k--) {
        int dim_idx = (k < axis) ? k : k + 1;
        index += (remainder % shape[dim_idx]) * strides[dim_idx];
        remainder /= shape[dim_idx];
      }
      index += i * axis_stride;
      sum += a[index];
    }
    means[tid] = sum / shape[axis];  // Store mean for this output position
  }
}

__global__ void __var_axis_kernel__(float* a, float* means, float* out, int* strides, int* shape, int axis, int ndim, int axis_stride, int size, int result_size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < result_size) {
    float variance = 0.0f;
    for (int i = 0; i < shape[axis]; i++) {
      int index = 0;
      int remainder = tid;
      for (int k = ndim - 2; k >= 0; k--) {
        int dim_idx = (k < axis) ? k : k + 1;
        index += (remainder % shape[dim_idx]) * strides[dim_idx];
        remainder /= shape[dim_idx];
      }
      index += i * axis_stride;
      float diff = a[index] - means[tid];
      variance += diff * diff;
    }
    out[tid] = variance;  // Store sum of squared differences
  }
}

__host__ void var_tensor_cuda(float* a, float* out, int axis, size_t size, int* strides, int* shape, size_t ndim, int ddof) {
  if (axis == -1) {
    // Global variance
    int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    float* temp_means;
    float* temp_vars;
    cudaMalloc(&temp_means, num_blocks * sizeof(float));
    cudaMalloc(&temp_vars, num_blocks * sizeof(float));
    
    // First pass: calculate mean
    __var_mean_kernel__<<<num_blocks, THREADS_PER_BLOCK>>>(a, temp_means, size);
    
    // Reduce means to single value
    int mean_blocks = num_blocks;
    while (mean_blocks > 1) {
      int num_blocks_next = (mean_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      float* next_temp;
      cudaMalloc(&next_temp, num_blocks_next * sizeof(float));
      __var_mean_kernel__<<<num_blocks_next, THREADS_PER_BLOCK>>>(temp_means, next_temp, mean_blocks);
      cudaFree(temp_means);
      temp_means = next_temp;
      mean_blocks = num_blocks_next;
    }

    // Divide by size to get actual mean
    float sum;
    cudaMemcpy(&sum, temp_means, sizeof(float), cudaMemcpyDeviceToHost);
    float mean = sum / size;
    cudaMemcpy(temp_means, &mean, sizeof(float), cudaMemcpyHostToDevice);

    // Second pass: calculate variance
    __var_kernel__<<<num_blocks, THREADS_PER_BLOCK>>>(a, temp_means, temp_vars, size);

    // Reduce variances to single value
    int var_blocks = num_blocks;
    while (var_blocks > 1) {
      int num_blocks_next = (var_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      float* next_temp;
      cudaMalloc(&next_temp, num_blocks_next * sizeof(float));
      __var_mean_kernel__<<<num_blocks_next, THREADS_PER_BLOCK>>>(temp_vars, next_temp, var_blocks);
      cudaFree(temp_vars);
      temp_vars = next_temp;
      var_blocks = num_blocks_next;
    }

    // Divide by (size - ddof) to get final variance
    float var_sum;
    cudaMemcpy(&var_sum, temp_vars, sizeof(float), cudaMemcpyDeviceToHost);
    int denominator = size - ddof;
    float variance = (denominator <= 0) ? 0.0f : var_sum / denominator;
    cudaMemcpy(out, &variance, sizeof(float), cudaMemcpyHostToDevice);

    cudaFree(temp_means);
    cudaFree(temp_vars);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
    cudaDeviceSynchronize();
  } else {
    // Axis-specific variance
    int axis_stride = strides[axis];
    int result_size = 1;
    for (int i = 0; i < ndim; i++) {
      if (i != axis) {
        result_size *= shape[i];
      }
    }

    int *d_strides, *d_shape;
    float *d_means;
    cudaMalloc(&d_strides, ndim * sizeof(int));
    cudaMalloc(&d_shape, ndim * sizeof(int));
    cudaMalloc(&d_means, result_size * sizeof(float));
    cudaMemcpy(d_strides, strides, ndim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, shape, ndim * sizeof(int), cudaMemcpyHostToDevice);
    
    int num_blocks = (result_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // First pass: calculate means
    __var_axis_mean_kernel__<<<num_blocks, THREADS_PER_BLOCK>>>(a, d_means, d_strides, d_shape, axis, ndim, axis_stride, size, result_size);

    // Second pass: calculate variances
    __var_axis_kernel__<<<num_blocks, THREADS_PER_BLOCK>>>(a, d_means, out, d_strides, d_shape, axis, ndim, axis_stride, size, result_size);

    // Divide by (axis_size - ddof) to get final variances
    int axis_size = shape[axis];
    int denominator = axis_size - ddof;
    if (denominator > 0) {
      int threads_per_block = min(result_size, THREADS_PER_BLOCK);
      int blocks = (result_size + threads_per_block - 1) / threads_per_block;
      __divide_kernel__<<<blocks, threads_per_block>>>(out, result_size, (float)denominator);
    } else {
      // Set all values to 0 if denominator <= 0
      cudaMemset(out, 0, result_size * sizeof(float));
    }

    cudaFree(d_strides);
    cudaFree(d_shape);
    cudaFree(d_means);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
    cudaDeviceSynchronize();
  }
}

// ====================== STD OPS ======================

__global__ void __std_finalize_kernel__(float* var, float* out, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    out[tid] = sqrtf(var[tid]);
  }
}

__host__ void std_tensor_cuda(float* a, float* out, int axis, size_t size, int* strides, int* shape, size_t ndim, int ddof) {
  if (axis == -1) {
    // Global standard deviation
    // First calculate variance
    var_tensor_cuda(a, out, axis, size, strides, shape, ndim, ddof);

    // Then take square root
    float variance;
    cudaMemcpy(&variance, out, sizeof(float), cudaMemcpyDeviceToHost);
    float std_dev = sqrtf(variance);
    cudaMemcpy(out, &std_dev, sizeof(float), cudaMemcpyHostToDevice);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
    cudaDeviceSynchronize();
  } else {
    // Axis-specific standard deviation
    int result_size = 1;
    for (int i = 0; i < ndim; i++) {
      if (i != axis) {
        result_size *= shape[i];
      }
    }

    // First calculate variance (reuses the variance calculation)
    var_tensor_cuda(a, out, axis, size, strides, shape, ndim, ddof);

    // Then take square root of each element
    int num_blocks = (result_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    __std_finalize_kernel__<<<num_blocks, THREADS_PER_BLOCK>>>(out, out, result_size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
    cudaDeviceSynchronize();
  }
}