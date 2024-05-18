#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tensor.h"

Tensor* create_tensor(int *data, int *shape, int ndim, int requires_grad) {
    Tensor *tensor = (Tensor*)malloc(sizeof(Tensor));
    tensor->data = data;
    tensor->shape = shape;
    tensor->ndim = ndim;
    tensor->requires_grad = requires_grad;
    tensor->grad = (int*)calloc(1, sizeof(int));
    return tensor;
}

void free_tensor(Tensor *tensor) {
    free(tensor->data);
    free(tensor->shape);
    free(tensor->grad);
    free(tensor);
}

void print_tensor(Tensor *tensor) {
    for (int i = 0; i < tensor->ndim; i++) {
        printf("%d ", tensor->data[i]);
    }
    printf("\n");
}

Tensor* tensor_add(Tensor *a, Tensor *b) {
    if (a->ndim != b->ndim) {
        printf("Shape mismatch\n");
        return NULL;
    }
    int *result_data = (int*)malloc(a->ndim * sizeof(int));
    for (int i = 0; i < a->ndim; i++) {
        result_data[i] = a->data[i] + b->data[i];
    }
    return create_tensor(result_data, a->shape, a->ndim, a->requires_grad || b->requires_grad);
}

Tensor* tensor_sub(Tensor *a, Tensor *b) {
    if (a->ndim != b->ndim) {
        printf("Shape mismatch\n");
        return NULL;
    }
    int *result_data = (int*)malloc(a->ndim * sizeof(int));
    for (int i = 0; i < a->ndim; i++) {
        result_data[i] = a->data[i] - b->data[i];
    }
    return create_tensor(result_data, a->shape, a->ndim, a->requires_grad || b->requires_grad);
}

Tensor* tensor_mul(Tensor *a, Tensor *b) {
    if (a->ndim != b->ndim) {
        printf("Shape mismatch\n");
        return NULL;
    }
    int *result_data = (int*)malloc(a->ndim * sizeof(int));
    for (int i = 0; i < a->ndim; i++) {
        result_data[i] = a->data[i] * b->data[i];
    }
    return create_tensor(result_data, a->shape, a->ndim, a->requires_grad || b->requires_grad);
}

Tensor* tensor_div(Tensor *a, Tensor *b) {
    if (a->ndim != b->ndim) {
        printf("Shape mismatch\n");
        return NULL;
    }
    int *result_data = (int*)malloc(a->ndim * sizeof(int));
    for (int i = 0; i < a->ndim; i++) {
        result_data[i] = a->data[i] / b->data[i];
    }
    return create_tensor(result_data, a->shape, a->ndim, a->requires_grad || b->requires_grad);
}

Tensor* tensor_neg(Tensor *a) {
    int *result_data = (int*)malloc(a->ndim * sizeof(int));
    for (int i = 0; i < a->ndim; i++) {
        result_data[i] = -a->data[i];
    }
    return create_tensor(result_data, a->shape, a->ndim, a->requires_grad);
}

Tensor* tensor_transpose(Tensor *a) {
    int rows = a->shape[0];
    int cols = a->shape[1];
    int *result_data = (int*)malloc(rows * cols * sizeof(int));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_data[j * rows + i] = a->data[i * cols + j];
        }
    }
    int *new_shape = (int*)malloc(2 * sizeof(int));
    new_shape[0] = cols;
    new_shape[1] = rows;
    return create_tensor(result_data, new_shape, 2, a->requires_grad);
}

Tensor* tensor_sum(Tensor *a) {
    int sum = 0;
    for (int i = 0; i < a->ndim; i++) {
        sum += a->data[i];
    }
    int *result_data = (int*)malloc(sizeof(int));
    result_data[0] = sum;
    int *new_shape = (int*)malloc(1 * sizeof(int));
    new_shape[0] = 1;
    return create_tensor(result_data, new_shape, 1, a->requires_grad);
}

int relu(int x) {
    return x > 0 ? x : 0;
}

Tensor* tensor_relu(Tensor *a) {
    int *result_data = (int*)malloc(a->ndim * sizeof(int));
    for (int i = 0; i < a->ndim; i++) {
        result_data[i] = relu(a->data[i]);
    }
    return create_tensor(result_data, a->shape, a->ndim, a->requires_grad);
}

int sigmoid(int x) {
    return 1 / (1 + exp(-x));
}

Tensor* tensor_sigmoid(Tensor *a) {
    int *result_data = (int*)malloc(a->ndim * sizeof(int));
    for (int i = 0; i < a->ndim; i++) {
        result_data[i] = sigmoid(a->data[i]);
    }
    return create_tensor(result_data, a->shape, a->ndim, a->requires_grad);
}

int tanh_func(int x) {
    return tanh(x);
}

Tensor* tensor_tanh(Tensor *a) {
    int *result_data = (int*)malloc(a->ndim * sizeof(int));
    for (int i = 0; i < a->ndim; i++) {
        result_data[i] = tanh_func(a->data[i]);
    }
    return create_tensor(result_data, a->shape, a->ndim, a->requires_grad);
}

int* flatten(int *arr, int size, int *new_size) {
    int *result = (int*)malloc(size * sizeof(int));
    int index = 0;
    for (int i = 0; i < size; i++) {
        result[index++] = arr[i];
    }
    *new_size = index;
    return result;
}

Tensor* tensor_flatten(Tensor *a) {
    int new_size;
    int *flattened_data = flatten(a->data, a->ndim, &new_size);
    int *new_shape = (int*)malloc(1 * sizeof(int));
    new_shape[0] = new_size;
    return create_tensor(flattened_data, new_shape, 1, a->requires_grad);
}