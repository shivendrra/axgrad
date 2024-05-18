#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    int *data;
    int *grad;
    int *shape;
    int ndim;
    int requires_grad;
} Tensor;

Tensor* create_tensor(int *data, int *shape, int ndim, int requires_grad);
void free_tensor(Tensor *tensor);
void print_tensor(Tensor *tensor);
Tensor* tensor_add(Tensor *a, Tensor *b);
Tensor* tensor_sub(Tensor *a, Tensor *b);
Tensor* tensor_mul(Tensor *a, Tensor *b);
Tensor* tensor_div(Tensor *a, Tensor *b);
Tensor* tensor_neg(Tensor *a);
Tensor* tensor_transpose(Tensor *a);
Tensor* tensor_sum(Tensor *a);
Tensor* tensor_relu(Tensor *a);
Tensor* tensor_sigmoid(Tensor *a);
Tensor* tensor_tanh(Tensor *a);
Tensor* tensor_flatten(Tensor *a);

#endif