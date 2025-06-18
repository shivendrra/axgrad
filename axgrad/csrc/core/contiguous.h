#ifndef __CONTIGUOUS__H__
#define __CONTIGUOUS__H__

#include "core.h"

extern "C" {
  int is_contiguous(Tensor* self); // checking if tensor is contiguous in memory
  void make_contiguous_inplace(Tensor* self);  // making tensor contiguous in-place (modifies original tensor)
  void contiguous_tensor_ops(void* src_data, void* dst_data, int* src_strides, int* shape, size_t ndim, size_t elem_size); // helper function for contiguous memory layout conversion
  // calculating flat index from multi-dimensional indices using strides
  size_t calculate_flat_index(int* indices, int* strides, size_t ndim);
  // converting flat index to multi-dimensional indices
  void flat_to_multi_index(size_t flat_idx, int* shape, size_t ndim, int* indices);
}


#endif  //!__CONTIGUOUS__H__