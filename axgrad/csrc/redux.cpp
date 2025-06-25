#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "redux_ops.h"
#include "cpu/ops_redux.h"

Tensor* sum_tensor(Tensor* a, int axis, bool keepdims) {
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }

  // validate axis
  if (axis != -1 && (axis < 0 || axis >= a->ndim)) {
    fprintf(stderr, "Error: axis %d out of range for tensor of dimension %zu\n", axis, a->ndim);
    exit(EXIT_FAILURE);
  }

  // calculate output shape and size
  int ndim;
  int* shape;
  size_t out_size;

  if (axis == -1) {
    // global sum - output is scalar (shape [1])
    ndim = 1;
    shape = (int*)malloc(1 * sizeof(int));
    if (shape == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(EXIT_FAILURE);
    }
    shape[0] = 1;
    out_size = 1;
  } else {
    // axis-specific sum - remove the specified axis
    ndim = a->ndim - 1;
    if (ndim == 0) {
      // if result would be 0-dimensional, make it 1-dimensional with size 1
      ndim = 1;
      shape = (int*)malloc(1 * sizeof(int));
      if (shape == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
      }
      shape[0] = 1;
      out_size = 1;
    } else {
      shape = (int*)malloc(ndim * sizeof(int));
      if (shape == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
      }
      for (int i = 0, j = 0; i < a->ndim; i++) {
        if (i != axis) {
          shape[j++] = a->shape[i];
        }
      }
      out_size = 1;
      for (int i = 0; i < ndim; i++) {
        out_size *= shape[i];
      }
    }
  }
  float* out = (float*)malloc(out_size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(shape);
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    free(shape);
    free(out);
    exit(EXIT_FAILURE);
  }

  sum_tensor_ops(a_float, out, a->shape, a->strides, a->size, shape, axis, a->ndim);
  if (keepdims) {
    free(shape);  // free the current shape
    // creating new shape with same number of dimensions as input
    ndim = a->ndim;
    shape = (int*)malloc(ndim * sizeof(int));
    if (shape == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      free(a_float);
      free(out);
      exit(EXIT_FAILURE);
    }
    if (axis == -1) {
      // all dimensions become 1
      for (int i = 0; i < ndim; i++) {
        shape[i] = 1;
      }
    } else {
      // copying original shape but set axis dimension to 1
      for (int i = 0; i < ndim; i++) {
        if (i == axis) {
          shape[i] = 1;
        } else {
          shape[i] = a->shape[i];
        }
      }
    }
  }

  // create result tensor
  dtype_t result_dtype = a->dtype;  // preserve original dtype
  Tensor* result = create_tensor(out, ndim, shape, out_size, result_dtype);
  free(a_float);
  free(out);
  if (shape) free(shape);
  return result;
}

Tensor* mean_tensor(Tensor* a, int axis, bool keepdims) {
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }

  // validate axis
  if (axis != -1 && (axis < 0 || axis >= a->ndim)) {
    fprintf(stderr, "Error: axis %d out of range for tensor of dimension %zu\n", axis, a->ndim);
    exit(EXIT_FAILURE);
  }

  // calculate output shape and size
  int ndim;
  int* shape;
  size_t out_size;

  if (axis == -1) {
    // global sum - output is scalar (shape [1])
    ndim = 1;
    shape = (int*)malloc(1 * sizeof(int));
    if (shape == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(EXIT_FAILURE);
    }
    shape[0] = 1;
    out_size = 1;
  } else {
    // axis-specific sum - remove the specified axis
    ndim = a->ndim - 1;
    if (ndim == 0) {
      // if result would be 0-dimensional, make it 1-dimensional with size 1
      ndim = 1;
      shape = (int*)malloc(1 * sizeof(int));
      if (shape == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
      }
      shape[0] = 1;
      out_size = 1;
    } else {
      shape = (int*)malloc(ndim * sizeof(int));
      if (shape == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
      }
      for (int i = 0, j = 0; i < a->ndim; i++) {
        if (i != axis) {
          shape[j++] = a->shape[i];
        }
      }
      out_size = 1;
      for (int i = 0; i < ndim; i++) {
        out_size *= shape[i];
      }
    }
  }
  float* out = (float*)malloc(out_size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(shape);
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    free(shape);
    free(out);
    exit(EXIT_FAILURE);
  }

  mean_tensor_ops(a_float, out, a->shape, a->strides, a->size, shape, axis, a->ndim);
  if (keepdims) {
    free(shape);  // free the current shape
    // creating new shape with same number of dimensions as input
    ndim = a->ndim;
    shape = (int*)malloc(ndim * sizeof(int));
    if (shape == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      free(a_float);
      free(out);
      exit(EXIT_FAILURE);
    }
    if (axis == -1) {
      // all dimensions become 1
      for (int i = 0; i < ndim; i++) {
        shape[i] = 1;
      }
    } else {
      // copying original shape but set axis dimension to 1
      for (int i = 0; i < ndim; i++) {
        if (i == axis) {
          shape[i] = 1;
        } else {
          shape[i] = a->shape[i];
        }
      }
    }
  }

  // create result tensor
  dtype_t result_dtype = a->dtype;  // preserve original dtype
  Tensor* result = create_tensor(out, ndim, shape, out_size, result_dtype);
  free(a_float);
  free(out);
  if (shape) free(shape);
  return result;
}

Tensor* max_tensor(Tensor* a, int axis, bool keepdims) {
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }

  // validate axis
  if (axis != -1 && (axis < 0 || axis >= a->ndim)) {
    fprintf(stderr, "Error: axis %d out of range for tensor of dimension %zu\n", axis, a->ndim);
    exit(EXIT_FAILURE);
  }

  // calculate output shape and size
  int ndim;
  int* shape;
  size_t out_size;

  if (axis == -1) {
    // global sum - output is scalar (shape [1])
    ndim = 1;
    shape = (int*)malloc(1 * sizeof(int));
    if (shape == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(EXIT_FAILURE);
    }
    shape[0] = 1;
    out_size = 1;
  } else {
    // axis-specific sum - remove the specified axis
    ndim = a->ndim - 1;
    if (ndim == 0) {
      // if result would be 0-dimensional, make it 1-dimensional with size 1
      ndim = 1;
      shape = (int*)malloc(1 * sizeof(int));
      if (shape == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
      }
      shape[0] = 1;
      out_size = 1;
    } else {
      shape = (int*)malloc(ndim * sizeof(int));
      if (shape == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
      }
      for (int i = 0, j = 0; i < a->ndim; i++) {
        if (i != axis) {
          shape[j++] = a->shape[i];
        }
      }
      out_size = 1;
      for (int i = 0; i < ndim; i++) {
        out_size *= shape[i];
      }
    }
  }
  float* out = (float*)malloc(out_size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(shape);
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    free(shape);
    free(out);
    exit(EXIT_FAILURE);
  }

  max_tensor_ops(a_float, out, a->size, a->shape, a->strides, shape, axis, a->ndim);
  if (keepdims) {
    free(shape);  // free the current shape
    // creating new shape with same number of dimensions as input
    ndim = a->ndim;
    shape = (int*)malloc(ndim * sizeof(int));
    if (shape == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      free(a_float);
      free(out);
      exit(EXIT_FAILURE);
    }
    if (axis == -1) {
      // all dimensions become 1
      for (int i = 0; i < ndim; i++) {
        shape[i] = 1;
      }
    } else {
      // copying original shape but set axis dimension to 1
      for (int i = 0; i < ndim; i++) {
        if (i == axis) {
          shape[i] = 1;
        } else {
          shape[i] = a->shape[i];
        }
      }
    }
  }

  // create result tensor
  dtype_t result_dtype = a->dtype;  // preserve original dtype
  Tensor* result = create_tensor(out, ndim, shape, out_size, result_dtype);
  free(a_float);
  free(out);
  if (shape) free(shape);
  return result;
}

Tensor* min_tensor(Tensor* a, int axis, bool keepdims) {
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }

  // validate axis
  if (axis != -1 && (axis < 0 || axis >= a->ndim)) {
    fprintf(stderr, "Error: axis %d out of range for tensor of dimension %zu\n", axis, a->ndim);
    exit(EXIT_FAILURE);
  }

  // calculate output shape and size
  int ndim;
  int* shape;
  size_t out_size;

  if (axis == -1) {
    // global sum - output is scalar (shape [1])
    ndim = 1;
    shape = (int*)malloc(1 * sizeof(int));
    if (shape == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(EXIT_FAILURE);
    }
    shape[0] = 1;
    out_size = 1;
  } else {
    // axis-specific sum - remove the specified axis
    ndim = a->ndim - 1;
    if (ndim == 0) {
      // if result would be 0-dimensional, make it 1-dimensional with size 1
      ndim = 1;
      shape = (int*)malloc(1 * sizeof(int));
      if (shape == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
      }
      shape[0] = 1;
      out_size = 1;
    } else {
      shape = (int*)malloc(ndim * sizeof(int));
      if (shape == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
      }
      for (int i = 0, j = 0; i < a->ndim; i++) {
        if (i != axis) {
          shape[j++] = a->shape[i];
        }
      }
      out_size = 1;
      for (int i = 0; i < ndim; i++) {
        out_size *= shape[i];
      }
    }
  }
  float* out = (float*)malloc(out_size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(shape);
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    free(shape);
    free(out);
    exit(EXIT_FAILURE);
  }

  min_tensor_ops(a_float, out, a->size, a->shape, a->strides, shape, axis, a->ndim);
  if (keepdims) {
    free(shape);  // free the current shape
    // creating new shape with same number of dimensions as input
    ndim = a->ndim;
    shape = (int*)malloc(ndim * sizeof(int));
    if (shape == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      free(a_float);
      free(out);
      exit(EXIT_FAILURE);
    }
    if (axis == -1) {
      // all dimensions become 1
      for (int i = 0; i < ndim; i++) {
        shape[i] = 1;
      }
    } else {
      // copying original shape but set axis dimension to 1
      for (int i = 0; i < ndim; i++) {
        if (i == axis) {
          shape[i] = 1;
        } else {
          shape[i] = a->shape[i];
        }
      }
    }
  }

  // create result tensor
  dtype_t result_dtype = a->dtype;  // preserve original dtype
  Tensor* result = create_tensor(out, ndim, shape, out_size, result_dtype);
  free(a_float);
  free(out);
  free(out);
  if (shape) free(shape);
  return result;
}

Tensor* var_tensor(Tensor* a, int axis, int ddof) {
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointer is null!\n");
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }

  // calculate result shape and size
  int* result_shape;
  size_t result_size;
  int result_ndim;
  if (axis == -1) {
    // global variance - scalar result
    result_ndim = 0;  // scalar
    result_shape = NULL;
    result_size = 1;
  } else {
    // axis-specific variance
    if (axis < 0 || axis >= a->ndim) {
      fprintf(stderr, "Invalid axis %d for tensor with %zu dimensions\n", axis, a->ndim);
      free(a_float);
      exit(EXIT_FAILURE);
    }
    result_ndim = a->ndim - 1;
    result_shape = (int*)malloc(result_ndim * sizeof(int));
    if (result_shape == NULL) {
      fprintf(stderr, "Memory allocation failed for result shape\n");
      free(a_float);
      exit(EXIT_FAILURE);
    }
    // copy shape excluding the axis dimension
    int j = 0;
    result_size = 1;
    for (int i = 0; i < a->ndim; i++) {
      if (i != axis) {
        result_shape[j] = a->shape[i];
        result_size *= a->shape[i];
        j++;
      }
    }
  }
  float* out = (float*)malloc(result_size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed for output\n");
    free(a_float);
    if (result_shape) free(result_shape);
    exit(EXIT_FAILURE);
  }

  var_tensor_ops(a_float, out, a->size, a->shape, a->strides, result_shape, axis, a->ndim, ddof);
  Tensor* result;
  if (axis == -1) {
    // for scalar result, create a 1D tensor with size 1
    int scalar_shape[1] = {1};
    result = create_tensor(out, 1, scalar_shape, 1, DTYPE_FLOAT32);
  } else {
    result = create_tensor(out, result_ndim, result_shape, result_size, DTYPE_FLOAT32);
  }

  free(a_float);
  free(out);
  if (result_shape) free(result_shape);
  return result;
}

Tensor* std_tensor(Tensor* a, int axis, int ddof) {
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointer is null!\n");
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }

  // calculate result shape and size
  int* result_shape;
  size_t result_size;
  int result_ndim;
  if (axis == -1) {
    // global standard deviation - scalar result
    result_ndim = 0;  // scalar
    result_shape = NULL;
    result_size = 1;
  } else {
    // axis-specific standard deviation
    if (axis < 0 || axis >= a->ndim) {
      fprintf(stderr, "Invalid axis %d for tensor with %zu dimensions\n", axis, a->ndim);
      free(a_float);
      exit(EXIT_FAILURE);
    }
    
    result_ndim = a->ndim - 1;
    result_shape = (int*)malloc(result_ndim * sizeof(int));
    if (result_shape == NULL) {
      fprintf(stderr, "Memory allocation failed for result shape\n");
      free(a_float);
      exit(EXIT_FAILURE);
    }
    int j = 0;
    result_size = 1;
    for (int i = 0; i < a->ndim; i++) {
      if (i != axis) {
        result_shape[j] = a->shape[i];
        result_size *= a->shape[i];
        j++;
      }
    }
  }
  float* out = (float*)malloc(result_size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed for output\n");
    free(a_float);
    if (result_shape) free(result_shape);
    exit(EXIT_FAILURE);
  }

  std_tensor_ops(a_float, out, a->size, a->shape, a->strides, result_shape, axis, a->ndim, ddof);
  Tensor* result;
  if (axis == -1) {
    // for scalar result, create a 1D tensor with size 1
    int scalar_shape[1] = {1};
    result = create_tensor(out, 1, scalar_shape, 1, DTYPE_FLOAT32);
  } else {
    result = create_tensor(out, result_ndim, result_shape, result_size, DTYPE_FLOAT32);
  }

  free(a_float);
  free(out);
  if (result_shape) free(result_shape);
  return result;
}