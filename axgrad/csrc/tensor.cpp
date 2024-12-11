#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "tensor.h"
#include "cpu.h"
#include "cuda.h"

Tensor* create_tensor(float* data, int* shape, int ndim, char* device) {
  Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
  if (tensor == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  tensor->data = data;
  tensor->shape = shape;
  tensor->ndim = ndim;
  self->device = (char*)malloc(strlen(device) + 1);
  if (device != NULL) {
    strcpy(self->device, device);
  } else {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }
  tensor->size = 1;
  for (int i = 0; i < ndim; i++) {
    tensor->size *= shape[i];
  }

  tensor->strides = (int*)malloc(ndim * sizeof(int));
  if (tensor->strides == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  int stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    tensor->strides[i] = stride;
    stride *= shape[i];
  }

  return tensor;
}

void to_device(Tensor* a, char* device) {
  int device_id = 0;
  char *end_ptr, *device_type;
  long num = strtol(device, &end_ptr, 10);
  if (*end_ptr == '\0') {
    device_id = (int)num;
    device_type = new char[strlen("cuda") + 1];
    strcpy(device_type, "cuda");
  } else {
    device_type = new char[strlen("cpu") + 1];
    strcpy(device_type, "cpu");
  }

  if((strcmp(device_type, "cuda") == 0) && (strcmp(a->device, "cpu") == 0)) {
    cpu_to_cuda(a, device_id);
  } else if ((strcmp(device_type, "cpu") == 0) && (strcmp(a->device, "cuda") == 0)) {
    cuda_to_cpu(a);
  }
  free(device_type);
}

void delete_tensor(Tensor* tensor) {
  if (tensor != NULL) {
    free(tensor);
    tensor = NULL;
  }
}

void delete_shape(Tensor* tensor) {
  if (tensor->shape != NULL) {
    free(tensor->shape);
    tensor->shape = NULL;
  }
}

void delete_data(Tensor* tensor) {
  if (tensor->data != NULL) {
    free(tensor->data);
    tensor->data = NULL;
  }
}

void delete_strides(Tensor* tensor) {
  if (tensor->strides != NULL) {
    free(tensor->strides);
    tensor->strides = NULL;
  }
}

float get_item(Tensor* tensor, int* indices) {
  int index = 0;
  for (int i = 0; i < tensor->ndim; i++) {
    index += indices[i] * tensor->strides[i];
  }
  float result;
  result = tensor->data[index];
  return result;
}

Tensor* add_tensor(Tensor* a, Tensor* b) {
  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "Tensors must have the same no of dims %d and %d for addition\n", a->ndim, b->ndim);
    exit(1);
  }
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    add_tensor_cpu(a, b, out);
    return create_tensor(out, a->shape, a->ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* sub_tensor(Tensor* a, Tensor* b) {
  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "Tensors must have the same no of dims %d and %d for subtraction\n", a->ndim, b->ndim);
    exit(1);
  }
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    sub_tensor_cpu(a, b, out);
    return create_tensor(out, a->shape, a->ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* elemwise_mul_tensor(Tensor* a, Tensor* b) {
  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "Tensors must have the same no of dims %d and %d for elementwise multiplication\n", a->ndim, b->ndim);
    exit(1);
  }
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    mul_tensor_cpu(a, b, out);
    return create_tensor(out, a->shape, a->ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* add_broadcasted_tensor(Tensor* a, Tensor* b) {
  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
  int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  int* broadcasted_shape = (int*)malloc(max_ndim * sizeof(int));
  if (broadcasted_shape == NULL) {
    fprintf(stderr, "Memory allocation failed");
    exit(1);
  }
  for (int i = 0; i < max_ndim; i++) {
    int dim1 = i < a->ndim ? a->shape[a->ndim - 1 -i] : 1, dim2 = i < b->ndim ? b->shape[b->ndim - 1 -i] : 1;
    if (dim1 != dim2 && dim1 != 1 && dim2 != 2) {
      fprintf(stderr, "shapes are not compatible for broadcasting\n");
      exit(1);
    }
    broadcasted_shape[max_ndim - 1 - i] = dim1 > dim2 ? dim1 : dim2;
  }
  int broadcasted_size = 1;
  for (int i = 0; i < max_ndim; i++) {
    broadcasted_size *= broadcasted_shape[i];
  }
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    add_broadcasted_tensor_cpu(a, b, out, broadcasted_shape, broadcasted_size);
    return create_tensor(out, broadcasted_shape, max_ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* sub_broadcasted_tensor(Tensor* a, Tensor* b) {
  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
  int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  int* broadcasted_shape = (int*)malloc(max_ndim * sizeof(int));
  if (broadcasted_shape == NULL) {
    fprintf(stderr, "Memory allocation failed");
    exit(1);
  }
  for (int i = 0; i < max_ndim; i++) {
    int dim1 = i < a->ndim ? a->shape[a->ndim - 1 -i] : 1, dim2 = i < b->ndim ? b->shape[b->ndim - 1 -i] : 1;
    if (dim1 != dim2 && dim1 != 1 && dim2 != 2) {
      fprintf(stderr, "shapes are not compatible for broadcasting\n");
      exit(1);
    }
    broadcasted_shape[max_ndim - 1 - i] = dim1 > dim2 ? dim1 : dim2;
  }
  int broadcasted_size = 1;
  for (int i = 0; i < max_ndim; i++) {
    broadcasted_size *= broadcasted_shape[i];
  }
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    sub_broadcasted_tensor_cpu(a, b, out, broadcasted_shape, broadcasted_size);
    return create_tensor(out, broadcasted_shape, max_ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* elemwise_mul_broadcasted_tensor(Tensor* a, Tensor* b) {
  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
  int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  int* broadcasted_shape = (int*)malloc(max_ndim * sizeof(int));
  if (broadcasted_shape == NULL) {
    fprintf(stderr, "Memory allocation failed");
    exit(1);
  }
  for (int i = 0; i < max_ndim; i++) {
    int dim1 = i < a->ndim ? a->shape[a->ndim - 1 -i] : 1, dim2 = i < b->ndim ? b->shape[b->ndim - 1 -i] : 1;
    if (dim1 != dim2 && dim1 != 1 && dim2 != 2) {
      fprintf(stderr, "shapes are not compatible for broadcasting\n");
      exit(1);
    }
    broadcasted_shape[max_ndim - 1 - i] = dim1 > dim2 ? dim1 : dim2;
  }
  int broadcasted_size = 1;
  for (int i = 0; i < max_ndim; i++) {
    broadcasted_size *= broadcasted_shape[i];
  }
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    mul_broadcasted_tensor_cpu(a, b, out, broadcasted_shape, broadcasted_size);
    return create_tensor(out, broadcasted_shape, max_ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* matmul_tensor(Tensor* a, Tensor* b) {
  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
  if (a->shape[1] != b->shape[0]) {
    fprintf(stderr, "Incompatible shapes for matrix multiplication %dx%d and %dx%d\n", a->shape[0], a->shape[1], b->shape[0], b->shape[1]);
    exit(1);
  }
  int ndim = a->ndim + b->ndim - 2;
  int* shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  for (int i = 0; i < a->ndim - 1; i++) {
    shape[i] = a->shape[i];
  }
  for (int i = a->ndim - 1; i < ndim; i++) {
    shape[i] = a->shape[i - a->ndim + 2];
  }
  int size = 1;
  for (int i = 0; i < ndim; i++) {
    size *= shape[i];
  }
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    matmul_tensor_cpu(a, b, out);
    return create_tensor(out, shape, ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* batched_matmul_tensor(Tensor* a, Tensor* b) {
  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
  if (a->shape[0] != b->shape[0]) {
    fprintf(stderr, "Incompatible shapes for batched multiplication %dx%d and %dx%d\n", a->shape[0], a->shape[1], b->shape[0], a->shape[1]);
    exit(1);
  }
  if (a->shape[2] != b->shape[1]) {
    fprintf(stderr, "Incompatible shapes for matrix multiplication %dx%d and %dx%d\n", a->shape[0], a->shape[1], b->shape[0], b->shape[1]);
    exit(1);
  }
  int ndim = 3, *shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  shape[0] = a->shape[0], shape[1] = a->shape[1], shape[2] = a->shape[2];
  int size = 1;
  for (int i = 0; i < ndim; i++) {
    size *= shape[i];
  }
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    batched_matmul_tensor_cpu(a, b, out);
    return create_tensor(out, shape, ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* broadcasted_batched_matmul_tensor_cpu(Tensor* a, Tensor* b) {
  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
  if (a->shape[1] != b->shape[1]) {
    fprintf(stderr, "Incompatible shapes for broadcasted batched matrix multiplication %dx%d and %dx%dx%d\n", a->shape[0], a->shape[1], b->shape[0], b->shape[1], b->shape[2]);
    exit(1);
  }
  int ndim = 3, *shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  shape[0] = a->shape[0], shape[1] = a->shape[1], shape[2] = a->shape[2];
  int size = 1;
  for (int i = 0; i < ndim; i++) {
    size *= shape[i];
  }
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    broadcasted_matmul_tensor_cpu(a, b, out);
    return create_tensor(out, shape, ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* tensor_div_tensor(Tensor* a, Tensor* b) {
    if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "Tensors must have the same no of dims %d and %d for elementwise multiplication\n", a->ndim, b->ndim);
    exit(1);
  }
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    div_tensor_cpu(a, b, out);
    return create_tensor(out, a->shape, a->ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* scalar_mul_tensor(Tensor* a, float b) {
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    scalar_mul_tensor_cpu(a, b, out);
    return create_tensor(out, a->shape, a->ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* tensor_div_scalar(Tensor* a, float b) {
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    tensor_div_scalar_cpu(a, b, out);
    return create_tensor(out, a->shape, a->ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* scalar_div_tensor(float a, Tensor* b) {
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    scalar_div_tensor_cpu(a, b, out);
    return create_tensor(out, a->shape, a->ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* tensor_pow_scalar(Tensor* a, float exp) {
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    tensor_pow_scalar_cpu(a, exp, out);
    return create_tensor(out, a->shape, a->ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* scalar_pow_tensor(float base, Tensor* a) {
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    scalar_pow_tensor_cpu(base, a, out);
    return create_tensor(out, a->shape, a->ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* log_tensor(Tensor* a) {
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    log_tensor_cpu(a, out);
    return create_tensor(out, a->shape, a->ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* sum_tensor(Tensor* a, int axis, bool keepdim) {
  int ndim, *shape;
  if (axis > a->ndim - 1) {
    fprintf(stderr, "Error: axis out of range, must be smaller then tensor dims %d < %d", axis, a->ndim);
    exit(1);
  }
  if (axis == -1) {
    shape = (int*)malloc(1 * sizeof(int));
    shape[0] = 1, ndim = 1;
  } else {
    shape = (int*)malloc((a->ndim - 1) * sizeof(int));
    for (int i = 0, j = 0; i < a->ndim; ++i) {
      if (i != axis) shape[j++] = a->shape[i];
    }
    ndim = a->ndim - 1;
  }
  int axis_size = 1;
  for (int i = 0; i < ndim; i++) {
    axis_size *= shape[i];
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed");
    exit(1);
  }
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed");
      exit(1);
    }
    sum_tensor_cpu(a, out, axis_size, shape, axis);
    if (keepdim) {
      if (axis == -1) {
        ndim = a->ndim, shape = (int*)malloc((a->ndim) * sizeof(int));
        for (int i = 0; i < a->size; i++) {
          shape[i] = 1;
        }
      } else {
        shape = (int*)malloc(a->ndim * sizeof(int));
        for (int i = 0; i < a->size; i++) {
          shape[i] = a->shape[i];
        }
        shape[axis] = 1, ndim = a->ndim;
      }
    }
    return create_tensor(out, a->shape, a->ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* max_tensor(Tensor* a, int axis, bool keepdim) {
  int ndim, *shape;
  if (axis > a->ndim - 1) {
    fprintf(stderr, "Error: axis out of range, must be smaller then tensor dims %d < %d", axis, a->ndim);
    exit(1);
  }
  if (axis == -1) {
    shape = (int*)malloc(1 * sizeof(int));
    shape[0] = 1, ndim = 1;
  } else {
    shape = (int*)malloc((a->ndim - 1) * sizeof(int));
    for (int i = 0, j = 0; i < a->ndim; ++i) {
      if (i != axis) shape[j++] = a->shape[i];
    }
    ndim = a->ndim - 1;
  }
  int axis_size = 1;
  for (int i = 0; i < ndim; i++) {
    axis_size *= shape[i];
  }
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed");
      exit(1);
    }
    max_tensor_cpu(a, out, axis_size, shape, axis);
    if (keepdim) {
      if (axis == -1) {
        ndim = a->ndim, shape = (int*)malloc((a->ndim) * sizeof(int));
        for (int i = 0; i < a->size; i++) {
          shape[i] = 1;
        }
      } else {
        shape = (int*)malloc(a->ndim * sizeof(int));
        for (int i = 0; i < a->size; i++) {
          shape[i] = a->shape[i];
        }
        shape[axis] = 1, ndim = a->ndim;
      }
    }
    return create_tensor(out, a->shape, a->ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* min_tensor(Tensor* a, int axis, bool keepdim) {
  int ndim, *shape;
  if (axis > a->ndim - 1) {
    fprintf(stderr, "Error: axis out of range, must be smaller then tensor dims %d < %d", axis, a->ndim);
    exit(1);
  }
  if (axis == -1) {
    shape = (int*)malloc(1 * sizeof(int));
    shape[0] = 1, ndim = 1;
  } else {
    shape = (int*)malloc((a->ndim - 1) * sizeof(int));
    for (int i = 0, j = 0; i < a->ndim; ++i) {
      if (i != axis) shape[j++] = a->shape[i];
    }
    ndim = a->ndim - 1;
  }
  int axis_size = 1;
  for (int i = 0; i < ndim; i++) {
    axis_size *= shape[i];
  }
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed");
      exit(1);
    }
    min_tensor_cpu(a, out, axis_size, shape, axis);
    if (keepdim) {
      if (axis == -1) {
        ndim = a->ndim, shape = (int*)malloc((a->ndim) * sizeof(int));
        for (int i = 0; i < a->size; i++) {
          shape[i] = 1;
        }
      } else {
        shape = (int*)malloc(a->ndim * sizeof(int));
        for (int i = 0; i < a->size; i++) {
          shape[i] = a->shape[i];
        }
        shape[axis] = 1, ndim = a->ndim;
      }
    }
    return create_tensor(out, a->shape, a->ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* sin_tensor(Tensor* a) {
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    sin_tensor_cpu(a, out);
    return create_tensor(out, a->shape, a->ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* cos_tensor(Tensor* a) {
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    cos_tensor_cpu(a, out);
    return create_tensor(out, a->shape, a->ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* sigmoid_tensor(Tensor* a) {
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    sigmoid_tensor_cpu(a, out);
    return create_tensor(out, a->shape, a->ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* tanh_tensor(Tensor* a) {
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    tanh_tensor_cpu(a, out);
    return create_tensor(out, a->shape, a->ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* relu_tensor(Tensor* a) {
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    relu_tensor_cpu(a, out);
    return create_tensor(out, a->shape, a->ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* reshape_tensor(Tensor* a, int* new_shape, int new_ndim) {
  int ndim = new_ndim, *shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed");
    exit(1);
  }
  for (int i = 0; i < ndim; i++) {
    shape[i] = new_shape[i];
  }
  int size = 1;
  for (int i = 0; i < new_ndim; i++) {
    size *= shape[i];
  }
  if (size != a->size) {
    fprintf(stderr, "Can't reshape the tensor. tensor's size doesn't match the target size: %d != %d", a->size, size);
  }
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed!");
      exit(1);
    }
    reassign_tensor_cpu(a, out);
    return create_tensor(out, shape, ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* transpose_tensor(Tensor* a) {
  int ndim = a->ndim, *shape = (int*)malloc(ndim * sizeof(int)), size = a->size;
    if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed");
    exit(1);
  }
  for (int i = 0; i < ndim; i++) {
    shape[i] = a->shape[ndim - 1 - i];
  }
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed!");
      exit(1);
    }
    switch(ndim) {
      case 1:
        transpose_1d_tensor_cpu(a, out);
        break;
      case 2:
        transpose_2d_tensor_cpu(a, out);
        break;
      case 3:
        transpose_3d_tensor_cpu(a, out);
        break;
      default:
        fprintf(stderr, "Transpose supported only for 3-dim tensor");
        exit(1);
    }
    return create_tensor(out, shape, ndim);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

void make_contiguous(Tensor* a) {
  int* new_strides = (int*)malloc(a->ndim * sizeof(int));
  if (new_strides == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
  }
  int stride = 1;
  for (int i = a->ndim - 1; i >= 0; i--) {
    new_strides[i] = stride;
    stride *= a->shape[i];
  }
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed!");
      exit(1);
    }
    make_contagious_tensor_cpu(a, out, new_strides);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* equal_tensor(Tensor* a, Tensor* b) {
  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "Tensors must have same dimensions %d and %d for equal", a->ndim, b->ndim);
    exit(1);
  }
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed!");
      exit(1);
    }
    equal_tensor_cpu(a, b, out);
    return create_tensor(out, a->shape, a->ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* equal_broadcasted_tensor(Tensor* a, Tensor* b) {
  if (strcmp(a->device, b->device) != 0) {
    fprintf(stderr, "Tensors must be on the same devices: %s and %s\n", a->device, b->device);
    exit(1);
  }
  int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  int* broadcasted_shape = (int*)malloc(max_ndim * sizeof(int));
  if (broadcasted_shape == NULL) {
    fprintf(stderr, "Memory allocation failed");
    exit(1);
  }
  for (int i = 0; i < max_ndim; i++) {
    int dim1 = i<a->ndim ? a->shape[a->ndim - max_ndim + i] : 1;
    int dim2 = i<b->ndim ? b->shape[b->ndim - max_ndim + i] : 1;
    if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
      fprintf(stderr, "Shapes are not compatible for broadcasting\n");
      exit(1);
    }
    broadcasted_shape[max_ndim - 1 - i] = dim1 > dim2 ? dim1 : dim2;
  }
  int broadcasted_size = 1;
  for (int i = 0; i < max_ndim; i++) {
    broadcasted_size *= broadcasted_shape[i];
  }
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed!");
      exit(1);
    }
    equal_broadcasted_tensor_cpu(a, b, out, broadcasted_shape, broadcasted_size);
    return create_tensor(out, broadcasted_shape, max_ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* zeros_like_tensor(Tensor* a) {
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    ones_like_tensor_cpu(a, out);
    return create_tensor(out, a->shape, a->ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}

Tensor* ones_like_tensor(Tensor* a) {
  if (strcmp(a->device, "cpu") == 0) {
    float* out = (float*)malloc(a->size * sizeof(float));
    if (out == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
    }
    ones_like_tensor_cpu(a, out);
    return create_tensor(out, a->shape, a->ndim, a->device);
  } else {
    ///////////////////////
    ///// placeholder /////
    ///////////////////////
  }
}