#include <stdio.h>
#include <stdlib.h>
#include "binary_ops.h"
#include "cpu/ops_binary.h"

Tensor* add_tensor(Tensor* a, Tensor* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "tensors must have the same no of dims %d and %d for addition\n", a->ndim, b->ndim);
    exit(EXIT_FAILURE);
  }

  // checking if shapes match
  for (size_t i = 0; i < a->ndim; i++) {
    if (a->shape[i] != b->shape[i]) {
      fprintf(stderr, "tensors must have the same shape for addition\n");
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
  add_ops(a_float, b_float, out, a->size);  // Perform the addition operation
  dtype_t result_dtype = promote_dtypes(a->dtype, b->dtype);  // determining result dtype
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(b_float);
  free(out);

  return result;
}

Tensor* add_scalar_tensor(Tensor* a, float b) {
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  // converting both tensors to float32 for computation
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  add_scalar_ops(a_float, b, out, a->size);  // Perform the addition operation
  dtype_t result_dtype = a->dtype;  // determining result dtype
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);

  return result;
}

Tensor* add_broadcasted_tensor(Tensor* a, Tensor* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  int* broadcasted_shape = (int*)malloc(max_ndim * sizeof(int));
  if (broadcasted_shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < max_ndim; i++) {
    int dim1 = i < a->ndim ? a->shape[a->ndim - 1 - i] : 1;
    int dim2 = i < b->ndim ? b->shape[b->ndim - 1 - i] : 1;
    if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
      fprintf(stderr, "shapes are not compatible for broadcasting\n");
      free(broadcasted_shape);
      exit(EXIT_FAILURE);
    }
    broadcasted_shape[max_ndim - 1 - i] = dim1 > dim2 ? dim1 : dim2;
  }

  // calculate broadcasted size
  size_t broadcasted_size = 1;
  for (int i = 0; i < max_ndim; i++) {
    broadcasted_size *= broadcasted_shape[i];
  }
  // convert both tensors to float32 for computation
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* b_float = convert_to_float32(b->data, b->dtype, b->size);
  if (a_float == NULL || b_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (b_float) free(b_float);
    free(broadcasted_shape);
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(broadcasted_size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    free(b_float);
    free(broadcasted_shape);
    exit(EXIT_FAILURE);
  }
  add_broadcasted_tensor_ops(a_float, b_float, out, broadcasted_shape, broadcasted_size, a->ndim, b->ndim, a->shape, b->shape);
  // determining result dtype using proper dtype promotion
  dtype_t result_dtype = promote_dtypes(a->dtype, b->dtype);
  Tensor* result = create_tensor(out, max_ndim, broadcasted_shape, broadcasted_size, result_dtype);
  free(a_float);
  free(b_float);
  free(out);
  free(broadcasted_shape);  
  return result;
}

Tensor* sub_tensor(Tensor* a, Tensor* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "tensors must have the same no of dims %d and %d for subtraction\n", a->ndim, b->ndim);
    exit(EXIT_FAILURE);
  }

  // checking if shapes match
  for (size_t i = 0; i < a->ndim; i++) {
    if (a->shape[i] != b->shape[i]) {
      fprintf(stderr, "tensors must have the same shape for subtraction\n");
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
  sub_ops(a_float, b_float, out, a->size);  // Perform the addition operation
  dtype_t result_dtype = promote_dtypes(a->dtype, b->dtype);  // determining result dtype
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(b_float);
  free(out);

  return result;
}

Tensor* sub_scalar_tensor(Tensor* a, float b) {
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  sub_scalar_ops(a_float, b, out, a->size);  // Perform the addition operation
  dtype_t result_dtype = a->dtype;  // determining result dtype
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);

  return result;
}

Tensor* sub_broadcasted_tensor(Tensor* a, Tensor* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  int* broadcasted_shape = (int*)malloc(max_ndim * sizeof(int));
  if (broadcasted_shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < max_ndim; i++) {
    int dim1 = i < a->ndim ? a->shape[a->ndim - 1 - i] : 1;
    int dim2 = i < b->ndim ? b->shape[b->ndim - 1 - i] : 1;
    if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
      fprintf(stderr, "shapes are not compatible for broadcasting\n");
      free(broadcasted_shape);
      exit(EXIT_FAILURE);
    }
    broadcasted_shape[max_ndim - 1 - i] = dim1 > dim2 ? dim1 : dim2;
  }

  // calculate broadcasted size
  size_t broadcasted_size = 1;
  for (int i = 0; i < max_ndim; i++) {
    broadcasted_size *= broadcasted_shape[i];
  }
  // convert both tensors to float32 for computation
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* b_float = convert_to_float32(b->data, b->dtype, b->size);
  if (a_float == NULL || b_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (b_float) free(b_float);
    free(broadcasted_shape);
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(broadcasted_size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    free(b_float);
    free(broadcasted_shape);
    exit(EXIT_FAILURE);
  }
  sub_broadcasted_tensor_ops(a_float, b_float, out, broadcasted_shape, broadcasted_size, a->ndim, b->ndim, a->shape, b->shape);
  // determining result dtype using proper dtype promotion
  dtype_t result_dtype = promote_dtypes(a->dtype, b->dtype);
  Tensor* result = create_tensor(out, max_ndim, broadcasted_shape, broadcasted_size, result_dtype);
  free(a_float);
  free(b_float);
  free(out);
  free(broadcasted_shape);  
  return result;
}

Tensor* mul_tensor(Tensor* a, Tensor* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "tensors must have the same no of dims %d and %d for multiplication\n", a->ndim, b->ndim);
    exit(EXIT_FAILURE);
  }

  // checking if shapes match
  for (size_t i = 0; i < a->ndim; i++) {
    if (a->shape[i] != b->shape[i]) {
      fprintf(stderr, "tensors must have the same shape for multiplication\n");
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
  mul_ops(a_float, b_float, out, a->size);  // Perform the addition operation
  dtype_t result_dtype = promote_dtypes(a->dtype, b->dtype);  // determining result dtype
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(b_float);
  free(out);

  return result;
}

Tensor* mul_scalar_tensor(Tensor* a, float b) {
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  mul_scalar_ops(a_float, b, out, a->size);  // Perform the addition operation
  dtype_t result_dtype = a->dtype;  // determining result dtype
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);

  return result;
}

Tensor* mul_broadcasted_tensor(Tensor* a, Tensor* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  int* broadcasted_shape = (int*)malloc(max_ndim * sizeof(int));
  if (broadcasted_shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < max_ndim; i++) {
    int dim1 = i < a->ndim ? a->shape[a->ndim - 1 - i] : 1;
    int dim2 = i < b->ndim ? b->shape[b->ndim - 1 - i] : 1;
    if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
      fprintf(stderr, "shapes are not compatible for broadcasting\n");
      free(broadcasted_shape);
      exit(EXIT_FAILURE);
    }
    broadcasted_shape[max_ndim - 1 - i] = dim1 > dim2 ? dim1 : dim2;
  }

  // calculate broadcasted size
  size_t broadcasted_size = 1;
  for (int i = 0; i < max_ndim; i++) {
    broadcasted_size *= broadcasted_shape[i];
  }
  // convert both tensors to float32 for computation
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* b_float = convert_to_float32(b->data, b->dtype, b->size);
  if (a_float == NULL || b_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (b_float) free(b_float);
    free(broadcasted_shape);
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(broadcasted_size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    free(b_float);
    free(broadcasted_shape);
    exit(EXIT_FAILURE);
  }
  mul_broadcasted_tensor_ops(a_float, b_float, out, broadcasted_shape, broadcasted_size, a->ndim, b->ndim, a->shape, b->shape);
  // determining result dtype using proper dtype promotion
  dtype_t result_dtype = promote_dtypes(a->dtype, b->dtype);
  Tensor* result = create_tensor(out, max_ndim, broadcasted_shape, broadcasted_size, result_dtype);
  free(a_float);
  free(b_float);
  free(out);
  free(broadcasted_shape);  
  return result;
}

Tensor* div_tensor(Tensor* a, Tensor* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "tensors must have the same no of dims %d and %d for division\n", a->ndim, b->ndim);
    exit(EXIT_FAILURE);
  }

  // checking if shapes match
  for (size_t i = 0; i < a->ndim; i++) {
    if (a->shape[i] != b->shape[i]) {
      fprintf(stderr, "tensors must have the same shape for division\n");
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
  div_ops(a_float, b_float, out, a->size);  // Perform the addition operation
  dtype_t result_dtype = promote_dtypes(a->dtype, b->dtype);  // determining result dtype
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(b_float);
  free(out);

  return result;
}

Tensor* div_scalar_tensor(Tensor* a, float b) {
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  div_scalar_ops(a_float, b, out, a->size);  // Perform the addition operation
  dtype_t result_dtype = a->dtype;  // determining result dtype
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);

  return result;
}

Tensor* div_broadcasted_tensor(Tensor* a, Tensor* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  int* broadcasted_shape = (int*)malloc(max_ndim * sizeof(int));
  if (broadcasted_shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < max_ndim; i++) {
    int dim1 = i < a->ndim ? a->shape[a->ndim - 1 - i] : 1;
    int dim2 = i < b->ndim ? b->shape[b->ndim - 1 - i] : 1;
    if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
      fprintf(stderr, "shapes are not compatible for broadcasting\n");
      free(broadcasted_shape);
      exit(EXIT_FAILURE);
    }
    broadcasted_shape[max_ndim - 1 - i] = dim1 > dim2 ? dim1 : dim2;
  }

  // calculate broadcasted size
  size_t broadcasted_size = 1;
  for (int i = 0; i < max_ndim; i++) {
    broadcasted_size *= broadcasted_shape[i];
  }
  // convert both tensors to float32 for computation
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* b_float = convert_to_float32(b->data, b->dtype, b->size);
  if (a_float == NULL || b_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (b_float) free(b_float);
    free(broadcasted_shape);
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(broadcasted_size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    free(b_float);
    free(broadcasted_shape);
    exit(EXIT_FAILURE);
  }
  div_broadcasted_tensor_ops(a_float, b_float, out, broadcasted_shape, broadcasted_size, a->ndim, b->ndim, a->shape, b->shape);
  // determining result dtype using proper dtype promotion
  dtype_t result_dtype = promote_dtypes(a->dtype, b->dtype);
  Tensor* result = create_tensor(out, max_ndim, broadcasted_shape, broadcasted_size, result_dtype);
  free(a_float);
  free(b_float);
  free(out);
  free(broadcasted_shape);  
  return result;
}

Tensor* pow_tensor(Tensor* a, float exp) {
  if (a == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }

  // converting tensor to float32 for computation
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  if (a_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }  
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    exit(EXIT_FAILURE);
  }

  pow_tensor_ops(a_float, exp, out, a->size);
  // for power operations, promote integer types to float
  // keeping existing float precision
  dtype_t result_dtype;
  if (is_integer_dtype(a->dtype) || a->dtype == DTYPE_BOOL) {
    result_dtype = DTYPE_FLOAT32;
  } else {
    result_dtype = a->dtype; // keeping float32 or float64
  }
  
  Tensor* result = create_tensor(out, a->ndim, a->shape, a->size, result_dtype);
  free(a_float);
  free(out);
  return result;
}

Tensor* pow_scalar(float a, Tensor* exp) {
  if (exp == NULL) {
    fprintf(stderr, "Tensor value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* exp_float = convert_to_float32(exp->data, exp->dtype, exp->size);
  if (exp_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }

  float* out = (float*)malloc(exp->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(exp_float);
    exit(EXIT_FAILURE);
  }
  pow_scalar_ops(a, exp_float, out, exp->size);
  // for power operations, promote integer types to float
  // keeping existing float precision
  dtype_t result_dtype;
  if (is_integer_dtype(exp->dtype) || exp->dtype == DTYPE_BOOL) {
    result_dtype = DTYPE_FLOAT32;
  } else {
    result_dtype = exp->dtype; // keeping float32 or float64
  }
  Tensor* result = create_tensor(out, exp->ndim, exp->shape, exp->size, result_dtype);
  free(exp_float);
  free(out);  
  return result;
}