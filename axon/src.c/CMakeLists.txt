cmake_minimum_required(VERSION 3.10)
project(axon_tensor)

set(CMAKE_C_STANDARD 99)

add_library(tensor tensor.c)

add_executable(main main.c)
target_link_libraries(main tensor)