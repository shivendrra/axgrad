#ifndef __TENSOR__H__
#define __TENSOR__H__

typedef struct Tensor {
  float* data;
  float* grad;
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
  float get_item(Tensor* tensor);
}

#endif  //__TENSOR__H__