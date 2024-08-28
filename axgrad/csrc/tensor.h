#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <numeric>
#include <stdexcept>
#include <algorithm>

class Tensor {
public:
  Tensor(const std::vector<float>& data, const std::vector<size_t>& shape);
    
  const std::vector<float>& get_data() const;
  const std::vector<size_t>& get_shape() const;
  size_t get_ndim() const;
  size_t get_size() const;
    
  float get_item(const std::vector<size_t>& indices) const;
  Tensor add(const Tensor& other) const;
  Tensor multiply(const Tensor& other) const;
  Tensor reshape(const std::vector<size_t>& new_shape) const;
    
private:
  std::vector<float> data;
  std::vector<size_t> shape;
  std::vector<size_t> strides;
  size_t ndim;
  size_t size;
    
  size_t calculate_size(const std::vector<size_t>& shape) const;
  std::vector<size_t> calculate_strides(const std::vector<size_t>& shape) const;
  void verify_data_size() const;
  size_t calculate_flat_index(const std::vector<size_t>& indices) const;
};

#endif