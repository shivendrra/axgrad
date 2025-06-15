#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "contiguous.h"
#include "../cpu/helpers.h"

int is_contiguous(Tensor* self) {
  if (self == NULL || self->ndim == 0) return 1;
  
  // calulating expected contiguous strides
  int expected_stride = 1;
  for (int i = self->ndim - 1; i >= 0; i--) {
    if (self->strides[i] != expected_stride) {
      return 0;
    }
    expected_stride *= self->shape[i];
  }
  return 1;
}

void contiguous_tensor_ops(void* src_data, void* dst_data, int* src_strides, int* shape, size_t ndim, size_t elem_size) {
  if (ndim == 0) return;

  size_t total_size = 1;
  for (size_t i = 0; i < ndim; i++) {
    total_size *= shape[i];
  }

  int* indices = (int*)malloc(ndim * sizeof(int));
  memset(indices, 0, ndim * sizeof(int));

  char* src = (char*)src_data;
  char* dst = (char*)dst_data;
  for (size_t flat_idx = 0; flat_idx < total_size; flat_idx++) {
    // calulating source offset using original strides
    size_t src_offset = 0;
    for (size_t dim = 0; dim < ndim; dim++) {
      src_offset += indices[dim] * src_strides[dim] * elem_size;
    }

    // Destination offset is simply flat_idx * elem_size (contiguous)
    size_t dst_offset = flat_idx * elem_size;
    memcpy(dst + dst_offset, src + src_offset, elem_size);      // copying element

    // increment indices (like odometer)
    int carry = 1;
    for (int dim = ndim - 1; dim >= 0 && carry; dim--) {
      indices[dim] += carry;
      if (indices[dim] >= shape[dim]) {
        indices[dim] = 0;
        carry = 1;
      } else {
        carry = 0;
      }
    }
  }
  
  free(indices);
}

void make_contiguous_inplace(Tensor* self) {
  if (self == NULL || is_contiguous(self)) return;
  
  size_t elem_size = get_dtype_size(self->dtype);
  void* new_data = malloc(self->size * elem_size);
  
  // rearranging data to contiguous layout
  contiguous_tensor_ops(self->data, new_data, self->strides, self->shape, self->ndim, elem_size);

  // freed old data and update
  free(self->data);
  self->data = new_data;

  // ipdated strides to be contiguous
  int stride = 1;
  for (int i = self->ndim - 1; i >= 0; i--) {
    self->strides[i] = stride;
    stride *= self->shape[i];
  }
  for (size_t i = 0; i < self->ndim; i++) {
    self->backstrides[self->ndim - 1 - i] = self->strides[i];
  }
  self->is_view = 0;  // no longer a view after making contiguous
}

size_t calulating_flat_index(int* indices, int* strides, size_t ndim) {
  size_t flat_idx = 0;
  for (size_t i = 0; i < ndim; i++) {
    flat_idx += indices[i] * strides[i];
  }
  return flat_idx;
}

void flat_to_multi_index(size_t flat_idx, int* shape, size_t ndim, int* indices) {
  for (int i = ndim - 1; i >= 0; i--) {
    indices[i] = flat_idx % shape[i];
    flat_idx /= shape[i];
  }
}