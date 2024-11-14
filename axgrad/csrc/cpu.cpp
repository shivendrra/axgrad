#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void add_tensor(Tensor* a, Tensor* b, float* out) {
  for (int i = 0; i < a->size; i++) {
    out[i] = a->data[i] + b->data[i];
  }
}

void sub_tensor(Tensor* a, Tensor* b, float* out) {
  for (int i = 0; i < a->size; i++) {
    out[i] = a->data[i] - b->data[i];
  }
}

void mul_tensor(Tensor* a, Tensor* b, float* out) {
  for (int i = 0; i < a->size; i++) {
    out[i] = a->data[i] * b->data[i];
  }
}