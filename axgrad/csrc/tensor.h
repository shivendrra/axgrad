/* 
  - tensor.h header file for tensor.cpp & Tensor
  - contains all the ops & functions for Tensor & python front-end
  - manages all the cpu & gpu functionality
  - compile it as:
    -- '.so': g++ -shared -fPIC -o libtensor.so tensor.cpp cpu.cpp
    -- '.dll': g++ -shared -o libtensor.dll tensor.cpp cpu.cpp
*/

#ifndef TENSOR_H
#define TENSOR_H

typedef struct Tensor {
  float* data;
  int* strides;
  int* shape;
  int ndim;
  int size;
} Tensor;

extern "C" {
  Tensor* create_tensor(float* data, int* shape, int ndim);
  void delete_tensor(Tensor* tensor);
  void delete_strides(Tensor* tensor);
  void delete_shape(Tensor* tensor);
  void delete_data(Tensor* tensor);

  Tensor* add_tensor(Tensor* a, Tensor* b);
  Tensor* sub_tensor(Tensor* a, Tensor* b);
  Tensor* elemwise_mul_tensor(Tensor* a, Tensor* b);
  Tensor* scalar_mul_tensor(Tensor* a, float b);
  Tensor* scalar_div_tensor(Tensor* a, float b);
  Tensor* tensor_pow_scalar(Tensor* a, float exponent);
  Tensor* scalar_pow_tensor(float base, Tensor* a);
}

#endif