#include <stdio.h>
#include <stdlib.h>
#include "cpu/ops_shape.h"
#include "shape_ops.h"

Tensor* transpose_tensor(Tensor* a) {
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }  
  
  int ndim = a->ndim;
  int* result_shape = (int*)malloc(ndim * sizeof(int));
  if (result_shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }

  // creating the result shape (reversed dimensions)
  for (int i = 0; i < ndim; i++) {
    result_shape[i] = a->shape[ndim - 1 - i];
  }
  
  // converting tensor to float32 for computation
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    free(result_shape);
    exit(EXIT_FAILURE);
  }  
  
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    free(result_shape);
    exit(EXIT_FAILURE);
  }

  // performing transpose based on dimensions
  // IMPORTANT: passing the ORIGINAL shape to transpose functions, not the result shape
  switch(ndim) {
    case 1:
      transpose_1d_tensor_ops(a_float, out, a->shape);  // using original shape
      break;
    case 2:
      transpose_2d_tensor_ops(a_float, out, a->shape);  // using original shape
      break;
    case 3:
      transpose_3d_tensor_ops(a_float, out, a->shape);  // using original shape
      break;
    default:
    if (ndim > 3) {
      transpose_ndim_tensor_ops(a_float, out, a->shape, a->ndim);
    } else {
      fprintf(stderr, "Transpose supported only for 1-3 dimensional tensors\n");
      free(a_float);
      free(out);
      free(result_shape);
      exit(EXIT_FAILURE);
    }
  }
  dtype_t result_dtype = a->dtype;
  Tensor* result = create_tensor(out, ndim, result_shape, a->size, result_dtype);
  free(a_float);
  free(out);
  free(result_shape);
  return result;
}

Tensor* reshape_tensor(Tensor* a, int* new_shape, int new_ndim) {
  if (a == NULL || new_shape == NULL) {
    fprintf(stderr, "Tensor or shape pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  int* shape = (int*)malloc(new_ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }

  // copying new shape and calculate new size
  size_t new_size = 1;
  for (int i = 0; i < new_ndim; i++) {
    shape[i] = new_shape[i];
    new_size *= shape[i];
  }
  if (new_size != a->size) {
    fprintf(stderr, "Can't reshape the tensor. tensor's size doesn't match the target size: %zu != %zu\n", a->size, new_size);
    free(shape);
    exit(EXIT_FAILURE);
  }

  // converting tensor to float32 for computation
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    free(shape);
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    free(shape);
    exit(EXIT_FAILURE);
  }
  // performing reshape (basically just copy data)
  reassign_tensor_ops(a_float, out, a->size);
  dtype_t result_dtype = a->dtype;    // reshaping preserves the original dtype
  Tensor* result = create_tensor(out, new_ndim, shape, a->size, result_dtype);
  free(a_float);
  free(out);
  free(shape);
  return result;
}

Tensor* equal_tensor(Tensor* a, Tensor* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "tensors must have same dimensions %d and %d for equal\n", a->ndim, b->ndim);
    exit(EXIT_FAILURE);
  }

  // checking if shapes match
  for (size_t i = 0; i < a->ndim; i++) {
    if (a->shape[i] != b->shape[i]) {
      fprintf(stderr, "tensors must have the same shape for comparison\n");
      exit(EXIT_FAILURE);
    }
  }
  // converting both tensors to float32 for computation
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* b_float = convert_to_float32(b->data, b->dtype, b->size);
  if (a_float == NULL || b_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (b_float) free(b_float);
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    free(b_float);
    exit(EXIT_FAILURE);
  }
  // perform the equality comparison
  equal_tensor_ops(a_float, b_float, out, a->size);
  // comparison operations always return boolean type
  dtype_t result_dtype = DTYPE_BOOL;
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(b_float);
  free(out);
  return result;
}

Tensor* not_equal_tensor(Tensor* a, Tensor* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "tensors must have same dimensions %d and %d for equal\n", a->ndim, b->ndim);
    exit(EXIT_FAILURE);
  }

  // checking if shapes match
  for (size_t i = 0; i < a->ndim; i++) {
    if (a->shape[i] != b->shape[i]) {
      fprintf(stderr, "tensors must have the same shape for comparison\n");
      exit(EXIT_FAILURE);
    }
  }
  // converting both tensors to float32 for computation
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* b_float = convert_to_float32(b->data, b->dtype, b->size);
  if (a_float == NULL || b_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (b_float) free(b_float);
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    free(b_float);
    exit(EXIT_FAILURE);
  }
  // perform the equality comparison
  not_equal_tensor_ops(a_float, b_float, out, a->size);
  // comparison operations always return boolean type
  dtype_t result_dtype = DTYPE_BOOL;
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(b_float);
  free(out);
  return result;
}

Tensor* greater_tensor(Tensor* a, Tensor* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "tensors must have same dimensions %d and %d for equal\n", a->ndim, b->ndim);
    exit(EXIT_FAILURE);
  }

  // checking if shapes match
  for (size_t i = 0; i < a->ndim; i++) {
    if (a->shape[i] != b->shape[i]) {
      fprintf(stderr, "tensors must have the same shape for comparison\n");
      exit(EXIT_FAILURE);
    }
  }
  // converting both tensors to float32 for computation
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* b_float = convert_to_float32(b->data, b->dtype, b->size);
  if (a_float == NULL || b_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (b_float) free(b_float);
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    free(b_float);
    exit(EXIT_FAILURE);
  }
  // perform the equality comparison
  greater_tensor_ops(a_float, b_float, out, a->size);
  // comparison operations always return boolean type
  dtype_t result_dtype = DTYPE_BOOL;
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(b_float);
  free(out);
  return result;
}

Tensor* greater_equal_tensor(Tensor* a, Tensor* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "tensors must have same dimensions %d and %d for equal\n", a->ndim, b->ndim);
    exit(EXIT_FAILURE);
  }

  // checking if shapes match
  for (size_t i = 0; i < a->ndim; i++) {
    if (a->shape[i] != b->shape[i]) {
      fprintf(stderr, "tensors must have the same shape for comparison\n");
      exit(EXIT_FAILURE);
    }
  }
  // converting both tensors to float32 for computation
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* b_float = convert_to_float32(b->data, b->dtype, b->size);
  if (a_float == NULL || b_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (b_float) free(b_float);
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    free(b_float);
    exit(EXIT_FAILURE);
  }
  // perform the equality comparison
  greater_equal_tensor_ops(a_float, b_float, out, a->size);
  // comparison operations always return boolean type
  dtype_t result_dtype = DTYPE_BOOL;
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(b_float);
  free(out);
  return result;
}

Tensor* smaller_tensor(Tensor* a, Tensor* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "tensors must have same dimensions %d and %d for equal\n", a->ndim, b->ndim);
    exit(EXIT_FAILURE);
  }

  // checking if shapes match
  for (size_t i = 0; i < a->ndim; i++) {
    if (a->shape[i] != b->shape[i]) {
      fprintf(stderr, "tensors must have the same shape for comparison\n");
      exit(EXIT_FAILURE);
    }
  }
  // converting both tensors to float32 for computation
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* b_float = convert_to_float32(b->data, b->dtype, b->size);
  if (a_float == NULL || b_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (b_float) free(b_float);
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    free(b_float);
    exit(EXIT_FAILURE);
  }
  // perform the equality comparison
  smaller_tensor_ops(a_float, b_float, out, a->size);
  // comparison operations always return boolean type
  dtype_t result_dtype = DTYPE_BOOL;
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(b_float);
  free(out);
  return result;
}

Tensor* smaller_equal_tensor(Tensor* a, Tensor* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "tensors must have same dimensions %d and %d for equal\n", a->ndim, b->ndim);
    exit(EXIT_FAILURE);
  }

  // checking if shapes match
  for (size_t i = 0; i < a->ndim; i++) {
    if (a->shape[i] != b->shape[i]) {
      fprintf(stderr, "tensors must have the same shape for comparison\n");
      exit(EXIT_FAILURE);
    }
  }
  // converting both tensors to float32 for computation
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* b_float = convert_to_float32(b->data, b->dtype, b->size);
  if (a_float == NULL || b_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (b_float) free(b_float);
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    free(b_float);
    exit(EXIT_FAILURE);
  }
  // perform the equality comparison
  smaller_equal_tensor_ops(a_float, b_float, out, a->size);
  // comparison operations always return boolean type
  dtype_t result_dtype = DTYPE_BOOL;
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(b_float);
  free(out);
  return result;
}

Tensor* squeeze_tensor(Tensor* a, int axis) {
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  int new_ndim = 0;
  int* temp_shape = (int*)malloc(a->ndim * sizeof(int));
  if (temp_shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }

  // if axis is -1, remove all dimensions of size 1
  if (axis == -1) {
    for (int i = 0; i < a->ndim; i++) {
      if (a->shape[i] != 1) {
        temp_shape[new_ndim] = a->shape[i];
        new_ndim++;
      }
    }
  } else {
    // validate axis
    if (axis < 0 || axis >= a->ndim) {
      fprintf(stderr, "axis %d is out of bounds for tensor of dimension %zu\n", axis, a->ndim);
      free(temp_shape);
      exit(EXIT_FAILURE);
    }
    if (a->shape[axis] != 1) {
      fprintf(stderr, "cannot select an axis to squeeze out which has size not equal to one\n");
      free(temp_shape);
      exit(EXIT_FAILURE);
    }
    // remove specific axis
    for (int i = 0; i < a->ndim; i++) {
      if (i != axis) {
        temp_shape[new_ndim] = a->shape[i];
        new_ndim++;
      }
    }
  }

  // handling edge case where all dimensions are squeezed out
  if (new_ndim == 0) {
    new_ndim = 1;
    temp_shape[0] = 1;
  }
  int* shape = (int*)malloc(new_ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(temp_shape);
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < new_ndim; i++) {
    shape[i] = temp_shape[i];
  }
  free(temp_shape);
  
  // converting tensor to float32 for computation
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    free(shape);
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    free(shape);
    exit(EXIT_FAILURE);
  }
  reassign_tensor_ops(a_float, out, a->size);  // performing squeeze (basically just copy data)
  dtype_t result_dtype = a->dtype;  // squeeze preserves the original dtype
  Tensor* result = create_tensor(out, new_ndim, shape, a->size, result_dtype);
  free(a_float);
  free(out);
  free(shape);
  return result;
}

Tensor* expand_dims_tensor(Tensor* a, int axis) {
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  
  int new_ndim = a->ndim + 1;
  if (axis < 0) {
    axis = new_ndim + axis;   // normalizing negative axis
  }
  // validating axis
  if (axis < 0 || axis >= new_ndim) {
    fprintf(stderr, "axis %d is out of bounds for tensor of dimension %d\n", axis, new_ndim);
    exit(EXIT_FAILURE);
  }
  int* shape = (int*)malloc(new_ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }

  // create new shape with expanded dimension
  int old_idx = 0;
  for (int i = 0; i < new_ndim; i++) {
    if (i == axis) {
      shape[i] = 1;  // insert new dimension of size 1
    } else {
      shape[i] = a->shape[old_idx];
      old_idx++;
    }
  }
  
  // converting tensor to float32 for computation
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    free(shape);
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    free(shape);
    exit(EXIT_FAILURE);
  }
  reassign_tensor_ops(a_float, out, a->size);   // performing expand_dims (basically just copy data)
  dtype_t result_dtype = a->dtype;  // expand_dims preserves the original dtype
  Tensor* result = create_tensor(out, new_ndim, shape, a->size, result_dtype);
  free(a_float);
  free(out);
  free(shape);
  return result;
}

Tensor* flatten_tensor(Tensor* a) {
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  int new_ndim = 1;
  int* shape = (int*)malloc(new_ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }

  shape[0] = a->size;   // flattened tensor has single dimension with size equal to total elements
  // converting tensor to float32 for computation
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    free(shape);
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    free(shape);
    exit(EXIT_FAILURE);
  }

  reassign_tensor_ops(a_float, out, a->size);  // performing flatten (basically just copy data)
  dtype_t result_dtype = a->dtype;  // flatten preserves the original dtype
  Tensor* result = create_tensor(out, new_ndim, shape, a->size, result_dtype);
  free(a_float);
  free(out);
  free(shape);
  return result;
}