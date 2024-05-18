#include <stdio.h>
#include "tensor.h"

int main() {
  int data1[] = {1, 2, 3, 4};
  int shape1[] = {2, 2};
  Tensor *tensor1 = create_tensor(data1, shape1, 4, 0);

  int data2[] = {5, 6, 7, 8};
  Tensor *tensor2 = create_tensor(data2, shape1, 4, 0);

  Tensor *result = tensor_add(tensor1, tensor2);
  print_tensor(result);

  free_tensor(tensor1);
  free_tensor(tensor2);
  free_tensor(result);

  return 0;
}