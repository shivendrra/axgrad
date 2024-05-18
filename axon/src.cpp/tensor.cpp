#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

using namespace std;

namespace py = pybind11;

// Helper function to get shape of a tensor
vector<int> get_shape(const vector<int>& data) {
  return { static_cast<int>(data.size()) };
}

vector<int> get_shape(const vector<vector<int>>& data) {
  vector<int> shape = { static_cast<int>(data.size()) };
  auto sub_shape = get_shape(data[0]);
  shape.insert(shape.end(), sub_shape.begin(), sub_shape.end());
  return shape;
}

// Helper function to create a tensor filled with zeros
vector<int> zeros(const vector<int>& shape) {
  return vector<int>(shape[0], 0);
}

vector<vector<int>> zeros(const vector<vector<int>>& shape) {
  return vector<vector<int>>(shape[0], zeros(shape[1]));
}

// Flatten a nested vector
void flatten(const vector<vector<int>>& arr, vector<int>& new_arr) {
  for (const auto& sub_arr : arr) {
    flatten(sub_arr, new_arr);
  }
}

void flatten(const vector<int>& arr, vector<int>& new_arr) {
  new_arr.insert(new_arr.end(), arr.begin(), arr.end());
}

class Tensor {
public:
    vector<vector<int>> data;
    vector<vector<int>> grad;
    vector<int> shape;
    bool requires_grad;
    
    Tensor(const vector<vector<int>>& data, bool requires_grad = false)
        : data(data), requires_grad(requires_grad) {
        shape = get_shape(data);
        grad = zeros(shape);
    }

    Tensor(const vector<int>& data, bool requires_grad = false)
        : Tensor(vector<vector<int>>{ data }, requires_grad) {}

    string repr() const {
        string data_str = "";
        for (const auto& row : data) {
            data_str += "\n\t";
            for (const auto& elem : row) {
                data_str += to_string(elem) + " ";
            }
        }
        return "Tensor(data=" + data_str + ")\n";
    }

    const vector<int>& operator[](int index) const {
        return data[index];
    }

    vector<int>& operator[](int index) {
        return data[index];
    }

    Tensor operator+(const Tensor& other) const {
        if (shape != other.shape) {
            throw invalid_argument("Arrays must be of same shape & size");
        }

        vector<vector<int>> result = data;
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < data[i].size(); ++j) {
                result[i][j] += other.data[i][j];
            }
        }
        return Tensor(result);
    }

    Tensor operator*(const Tensor& other) const {
        if (shape != other.shape) {
            throw invalid_argument("Arrays must be of same shape & size");
        }

        vector<vector<int>> result = data;
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < data[i].size(); ++j) {
                result[i][j] *= other.data[i][j];
            }
        }
        return Tensor(result);
    }

    Tensor operator-() const {
        vector<vector<int>> result = data;
        for (auto& row : result) {
            for (auto& elem : row) {
                elem = -elem;
            }
        }
        return Tensor(result);
    }

    Tensor operator-(const Tensor& other) const {
        return *this + (-other);
    }

    Tensor operator/(const Tensor& other) const {
        vector<vector<int>> result = data;
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < data[i].size(); ++j) {
                result[i][j] /= other.data[i][j];
            }
        }
        return Tensor(result);
    }

    Tensor transpose() const {
        size_t rows = data.size();
        size_t cols = data[0].size();
        vector<vector<int>> result(cols, vector<int>(rows));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[j][i] = data[i][j];
            }
        }
        return Tensor(result);
    }

    vector<int> flatten() const {
        vector<int> new_arr;
        ::flatten(data, new_arr);
        return new_arr;
    }

    int sum() const {
        vector<int> flat = flatten();
        return accumulate(flat.begin(), flat.end(), 0);
    }

};

PYBIND11_MODULE(tensor_module, m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const vector<vector<int>>&, bool>(), py::arg("data"), py::arg("requires_grad") = false)
        .def(py::init<const vector<int>&, bool>(), py::arg("data"), py::arg("requires_grad") = false)
        .def("repr", &Tensor::repr)
        .def("__repr__", &Tensor::repr)
        .def("__getitem__", (const vector<int>& (Tensor::*)(int) const) &Tensor::operator[])
        .def("__setitem__", (vector<int>& (Tensor::*)(int)) &Tensor::operator[])
        .def("__add__", &Tensor::operator+)
        .def("__mul__", &Tensor::operator*)
        .def("__neg__", &Tensor::operator-)
        .def("__sub__", &Tensor::operator-)
        .def("__truediv__", &Tensor::operator/)
        .def("transpose", &Tensor::transpose)
        .def("flatten", &Tensor::flatten)
        .def("sum", &Tensor::sum)
        .def("broadcast", &Tensor::broadcast);
}
