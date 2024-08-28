#include "tensor.h"

Tensor::Tensor(const std::vector<float>& data, const std::vector<size_t>& shape)
  : data(data), shape(shape), strides(calculate_strides(shape)), ndim(shape.size()), size(calculate_size(shape)) {
  verify_data_size();
}

size_t Tensor::calculate_size(const std::vector<size_t>& shape) const {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
}

std::vector<size_t> Tensor::calculate_strides(const std::vector<size_t>& shape) const {
  std::vector<size_t> strides(shape.size());
  strides.back() = 1;
  for (int i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

void Tensor::verify_data_size() const {
  if (data.size() != size) {
    throw std::runtime_error("Data size does not match shape size");
  }
}

size_t Tensor::calculate_flat_index(const std::vector<size_t>& indices) const {
  if (indices.size() != ndim) {
    throw std::runtime_error("Number of indices does not match tensor dimensions");
  }
  size_t index = 0;
  for (size_t i = 0; i < ndim; ++i) {
    index += indices[i] * strides[i];
  }
  return index;
}

const std::vector<float>& Tensor::get_data() const {
  return data;
}

const std::vector<size_t>& Tensor::get_shape() const {
  return shape;
}

size_t Tensor::get_ndim() const {
  return ndim;
}

size_t Tensor::get_size() const {
  return size;
}

float Tensor::get_item(const std::vector<size_t>& indices) const {
  size_t index = calculate_flat_index(indices);
  return data[index];
}

Tensor Tensor::add(const Tensor& other) const {
  if (shape != other.shape) {
    throw std::runtime_error("Shapes do not match for addition");
  }
  std::vector<float> result(data.size());
  std::transform(data.begin(), data.end(), other.data.begin(), result.begin(), std::plus<float>());
  return Tensor(result, shape);
}

Tensor Tensor::multiply(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::runtime_error("Shapes do not match for multiplication");
    }
    std::vector<float> result(data.size());
    std::transform(data.begin(), data.end(), other.data.begin(), result.begin(), std::multiplies<float>());
    return Tensor(result, shape);
}

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_size = calculate_size(new_shape);
    if (new_size != size) {
        throw std::runtime_error("New shape size does not match data size");
    }
    return Tensor(data, new_shape);
}