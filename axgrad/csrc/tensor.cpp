#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

Tensor* create_tensor(float* data, int* shape, int ndim) {
  Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
  if (tensor == NULL) {
    fprintf(stderr, "Memmory allocation failed\n");
    exit(1);
  }
}