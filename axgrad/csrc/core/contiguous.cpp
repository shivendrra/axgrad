#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <immintrin.h>
#include "contiguous.h"
#include "../cpu/helpers.h"

#pragma GCC target("avx2,fma")
#pragma GCC optimize("O3")

// is_contiguous

int is_contiguous(Tensor* self) {
  if (!self || self->ndim == 0) return 1;
  int expected = 1;
  for (int i = self->ndim - 1; i >= 0; i--) {
    if (self->strides[i] != expected) return 0;
    expected *= self->shape[i];
  }
  return 1;
}

// flat_to_multi_index / flat index helpers

void flat_to_multi_index(size_t flat, int* shape, size_t ndim, int* idx) {
  for (int i = (int)ndim - 1; i >= 0; i--) {
    idx[i] = (int)(flat % shape[i]);
    flat   /= shape[i];
  }
}

size_t calulating_flat_index(int* indices, int* strides, size_t ndim) {
  size_t flat = 0;
  for (size_t i = 0; i < ndim; i++) flat += indices[i] * strides[i];
  return flat;
}

// contiguous_tensor_ops
// Replaces the serial odometer with a parallel flat-index loop.
// For float32 (elem_size==4) the inner copy uses AVX2 bulk loads when strides
// happen to be contiguous on the last dimension.

void contiguous_tensor_ops(void* src_data, void* dst_data, int* src_strides, int* shape, size_t ndim, size_t elem_size) {
  if (ndim == 0) return;

  size_t total = 1;
  for (size_t i = 0; i < ndim; i++) total *= shape[i];

  char* src = (char*)src_data;
  char* dst = (char*)dst_data;

  // Fast path: float32, last-dim stride == 1 → AVX2 row copies
  if (elem_size == 4 && ndim >= 1 && src_strides[ndim - 1] == 1) {
    size_t inner    = shape[ndim - 1];          // contiguous inner dim
    size_t outer    = total / inner;             // number of rows

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t row = 0; row < outer; row++) {
      // compute source row offset from the outer indices
      size_t tmp = row, src_row_off = 0;
      for (int d = (int)ndim - 2; d >= 0; d--) {
        int coord  = (int)(tmp % shape[d]); tmp /= shape[d];
        src_row_off += coord * src_strides[d];
      }
      float* s = (float*)src + src_row_off;
      float* d_ptr = (float*)dst + row * inner;

      size_t j = 0;
      for (; j + 8 <= inner; j += 8)
        _mm256_storeu_ps(d_ptr + j, _mm256_loadu_ps(s + j));
      for (; j < inner; j++) d_ptr[j] = s[j];
    }
    return;
  }

  // General path: arbitrary elem_size, parallel over flat output index
#ifdef _OPENMP
  #pragma omp parallel
  {
    int* idx = (int*)malloc(ndim * sizeof(int));
    #pragma omp for schedule(static)
    for (size_t fi = 0; fi < total; fi++) {
      flat_to_multi_index(fi, shape, ndim, idx);
      size_t src_off = 0;
      for (size_t d = 0; d < ndim; d++) src_off += idx[d] * src_strides[d];
      memcpy(dst + fi * elem_size, src + src_off * elem_size, elem_size);
    }
    free(idx);
  }
#else
  int* idx = (int*)malloc(ndim * sizeof(int));
  for (size_t fi = 0; fi < total; fi++) {
    flat_to_multi_index(fi, shape, ndim, idx);
    size_t src_off = 0;
    for (size_t d = 0; d < ndim; d++) src_off += idx[d] * src_strides[d];
    memcpy(dst + fi * elem_size, src + src_off * elem_size, elem_size);
  }
  free(idx);
#endif
}

// make_contiguous_inplace

void make_contiguous_inplace(Tensor* self) {
  if (!self || is_contiguous(self)) return;

  size_t elem_size = get_dtype_size(self->dtype);
  void*  new_data  = malloc(self->size * elem_size);
  if (!new_data) { fprintf(stderr, "malloc failed in make_contiguous_inplace\n"); return; }

  contiguous_tensor_ops(self->data, new_data, self->strides, self->shape, self->ndim, elem_size);

  free(self->data);
  self->data = new_data;

  // Update strides to row-major contiguous — AVX2 not applicable (tiny loop)
  int stride = 1;
  for (int i = self->ndim - 1; i >= 0; i--) {
    self->strides[i] = stride;
    stride *= self->shape[i];
  }
  self->is_view = 0;
}