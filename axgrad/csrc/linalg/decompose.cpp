#include <stdio.h>
#include <stdlib.h>
#include "../cpu/ops_decomp.h"
#include "decompose.h"

Tensor* det_tensor(Tensor* a) {
  if (a == NULL) {
    fprintf(stderr, "Tensors cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 2) {
    fprintf(stderr, "Only 2D tensor supported for det()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[0] != a->shape[1]) {
    fprintf(stderr, "Tensor must be square to compute det(). dim0 '%d' != dim1 '%d'\n", a->shape[0], a->shape[1]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(1 * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  shape[0] = 1;
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* out = (float*)malloc(1 * sizeof(float));
  if (a_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (out) free(out);
    if (shape) free(shape);
    exit(EXIT_FAILURE);
  }

  // Passing matrix dimension (shape[0]), not total size
  det_ops_tensor(a_float, out, a->shape[0]);
  Tensor* result = create_tensor(out, 1, shape, 1, a->dtype);
  free(a_float); 
  free(out); 
  free(shape);
  return result;
}

Tensor* batched_det_tensor(Tensor* a) {
  if (a == NULL) {
    fprintf(stderr, "Tensors cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 3) {
    fprintf(stderr, "Only 3D tensor supported for batched det()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[1] != a->shape[2]) {
    fprintf(stderr, "Tensor must be square to compute det(). dim1 '%d' != dim2 '%d'\n", a->shape[1], a->shape[2]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(1 * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  shape[0] = a->shape[0]; // Output should have batch size
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* out = (float*)malloc(a->shape[0] * sizeof(float)); // allocating for batch size
  if (a_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (out) free(out);
    if (shape) free(shape);
    exit(EXIT_FAILURE);
  }

  // Pass matrix dimension (shape[1])
  batched_det_ops(a_float, out, a->shape[1], a->shape[0]);
  Tensor* result = create_tensor(out, 1, shape, a->shape[0], a->dtype);
  free(a_float); 
  free(out); 
  free(shape);
  return result;
}

Tensor* eig_tensor(Tensor* a) {
  if (a == NULL) {
    fprintf(stderr, "Tensors cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 2) {
    fprintf(stderr, "Only 2D tensor supported for eig()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[0] != a->shape[1]) {
    fprintf(stderr, "Tensor must be square to compute eigenvalues. dim0 '%d' != dim1 '%d'\n", a->shape[0], a->shape[1]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(1 * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  shape[0] = a->shape[0]; // eigenvalues count equals matrix dimension
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->shape[0] * sizeof(float));
  if (a_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (out) free(out);
    if (shape) free(shape);
    exit(EXIT_FAILURE);
  }
  eigenvals_ops_tensor(a_float, out, a->shape[0]);
  Tensor* result = create_tensor(out, 1, shape, a->shape[0], a->dtype);
  free(a_float); 
  free(out); 
  free(shape);
  return result;
}

Tensor* eigv_tensor(Tensor* a) {
  if (a == NULL) {
    fprintf(stderr, "Tensors cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 2) {
    fprintf(stderr, "Only 2D tensor supported for eigv()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[0] != a->shape[1]) {
    fprintf(stderr, "Tensor must be square to compute eigenvectors. dim0 '%d' != dim1 '%d'\n", a->shape[0], a->shape[1]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(2 * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  shape[0] = a->shape[0]; shape[1] = a->shape[1]; // same dimensions as input matrix
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->size * sizeof(float));
  if (a_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (out) free(out);
    if (shape) free(shape);
    exit(EXIT_FAILURE);
  }
  eigenvecs_ops_tensor(a_float, out, a->shape[0]);
  Tensor* result = create_tensor(out, 2, shape, a->size, a->dtype);
  free(a_float); 
  free(out); 
  free(shape);
  return result;
}

Tensor* eigh_tensor(Tensor* a) {
  if (a == NULL) {
    fprintf(stderr, "Tensors cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 2) {
    fprintf(stderr, "Only 2D tensor supported for eigh()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[0] != a->shape[1]) {
    fprintf(stderr, "Tensor must be square to compute hermitian eigenvalues. dim0 '%d' != dim1 '%d'\n", a->shape[0], a->shape[1]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(1 * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  shape[0] = a->shape[0]; // eigenvalues count equals matrix dimension
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->shape[0] * sizeof(float));
  if (a_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (out) free(out);
    if (shape) free(shape);
    exit(EXIT_FAILURE);
  }

  eigenvals_h_ops_tensor(a_float, out, a->shape[0]);
  Tensor* result = create_tensor(out, 1, shape, a->shape[0], a->dtype);
  free(a_float); 
  free(out); 
  free(shape);
  return result;
}

Tensor* eighv_tensor(Tensor* a) {
  if (a == NULL) {
    fprintf(stderr, "Tensors cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 2) {
    fprintf(stderr, "Only 2D tensor supported for eighv()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[0] != a->shape[1]) {
    fprintf(stderr, "Tensor must be square to compute hermitian eigenvectors. dim0 '%d' != dim1 '%d'\n", a->shape[0], a->shape[1]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(2 * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  shape[0] = a->shape[0]; shape[1] = a->shape[1]; // same dimensions as input matrix
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->size * sizeof(float));
  if (a_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (out) free(out);
    if (shape) free(shape);
    exit(EXIT_FAILURE);
  }
  eigenvecs_h_ops_tensor(a_float, out, a->shape[0]);
  Tensor* result = create_tensor(out, 2, shape, a->size, a->dtype);
  free(a_float); 
  free(out); 
  free(shape);
  return result;
}

Tensor* batched_eig_tensor(Tensor* a) {
  if (a == NULL) {
    fprintf(stderr, "Tensors cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 3) {
    fprintf(stderr, "Only 3D tensor supported for batched eig()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[1] != a->shape[2]) {
    fprintf(stderr, "Tensor must be square to compute eigenvalues. dim1 '%d' != dim2 '%d'\n", a->shape[1], a->shape[2]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(2 * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  shape[0] = a->shape[0], shape[1] = a->shape[1];
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->shape[0] * a->shape[1] * sizeof(float));
  if (a_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (out) free(out);
    if (shape) free(shape);
    exit(EXIT_FAILURE);
  }

  batched_eigenvals_ops(a_float, out, a->shape[1], a->shape[0]);
  Tensor* result = create_tensor(out, 2, shape, a->shape[0] * a->shape[1], a->dtype);
  free(a_float); 
  free(out); 
  free(shape);
  return result;
}

Tensor* batched_eigv_tensor(Tensor* a) {
  if (a == NULL) {
    fprintf(stderr, "Tensors cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 3) {
    fprintf(stderr, "Only 3D tensor supported for batched eigv()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[1] != a->shape[2]) {
    fprintf(stderr, "Tensor must be square to compute eigenvectors. dim1 '%d' != dim2 '%d'\n", a->shape[1], a->shape[2]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(3 * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  shape[0] = a->shape[0], shape[1] = a->shape[1], shape[2] = a->shape[2];
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->size * sizeof(float));
  if (a_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (out) free(out);
    if (shape) free(shape);
    exit(EXIT_FAILURE);
  }
  batched_eigenvecs_ops(a_float, out, a->shape[1], a->shape[0]);
  Tensor* result = create_tensor(out, 3, shape, a->size, a->dtype);
  free(a_float); 
  free(out); 
  free(shape);
  return result;
}

Tensor* batched_eigh_tensor(Tensor* a) {
  if (a == NULL) {
    fprintf(stderr, "Tensors cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 3) {
    fprintf(stderr, "Only 3D tensor supported for batched eigh()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[1] != a->shape[2]) {
    fprintf(stderr, "Tensor must be square to compute hermitian eigenvalues. dim1 '%d' != dim2 '%d'\n", a->shape[1], a->shape[2]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(2 * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  shape[0] = a->shape[0], shape[1] = a->shape[1];
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->shape[0] * a->shape[1] * sizeof(float));
  if (a_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (out) free(out);
    if (shape) free(shape);
    exit(EXIT_FAILURE);
  }
  batched_eigenvals_h_ops(a_float, out, a->shape[1], a->shape[0]);
  Tensor* result = create_tensor(out, 2, shape, a->shape[0] * a->shape[1], a->dtype);
  free(a_float); 
  free(out); 
  free(shape);
  return result;
}

Tensor* batched_eighv_tensor(Tensor* a) {
  if (a == NULL) {
    fprintf(stderr, "Tensors cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 3) {
    fprintf(stderr, "Only 3D tensor supported for batched eighv()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[1] != a->shape[2]) {
    fprintf(stderr, "Tensor must be square to compute hermitian eigenvectors. dim1 '%d' != dim2 '%d'\n", a->shape[1], a->shape[2]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(3 * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  shape[0] = a->shape[0], shape[1] = a->shape[1], shape[2] = a->shape[2];
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->size * sizeof(float));
  if (a_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (out) free(out);
    if (shape) free(shape);
    exit(EXIT_FAILURE);
  }

  batched_eigenvecs_h_ops(a_float, out, a->shape[1], a->shape[0]);
  Tensor* result = create_tensor(out, 3, shape, a->size, a->dtype);
  free(a_float); 
  free(out); 
  free(shape);
  return result;
}