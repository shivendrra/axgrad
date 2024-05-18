cmake_minimum_required(VERSION 3.12)
project(tensor_module)

set(CMAKE_CXX_STANDARD 14)

find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Add the source file for the tensor module
pybind11_add_module(Axon axon/csrc/tensor.cpp)