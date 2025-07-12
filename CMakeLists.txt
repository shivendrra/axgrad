cmake_minimum_required(VERSION 3.18)

# Force Ninja generator on Windows
if(WIN32)
  set(CMAKE_GENERATOR "Ninja" CACHE INTERNAL "")
endif()

project(axgrad LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# For MinGW on Windows
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

file(GLOB_RECURSE CSRC_FILES "axgrad/csrc/*.c" "axgrad/csrc/*.cpp")
file(GLOB_RECURSE INC_FILES "axgrad/csrc/inc/*.h" "axgrad/csrc/inc/*.hpp")
file(GLOB_RECURSE BACK_FILES "axgrad/autograd/csrc/*.c" "axgrad/autograd/csrc/*.cpp" "axgrad/autograd/csrc/*.h" "axgrad/autograd/csrc/*.hpp")

message(STATUS "Found source files: ${CSRC_FILES}")
message(STATUS "Found header files: ${INC_FILES}")
message(STATUS "Found autograd files: ${BACK_FILES}")

if(NOT CSRC_FILES)
  message(FATAL_ERROR "No source files found in axgrad/csrc/")
endif()
if(NOT INC_FILES)
  message(FATAL_ERROR "No header files found in axgrad/csrc/inc")
endif()
if(NOT BACK_FILES)
  message(FATAL_ERROR "No source files found in axgrad/autograd/csrc")
endif()

add_library(tensor SHARED ${CSRC_FILES} ${BACK_FILES})
target_include_directories(tensor PRIVATE axgrad/csrc/inc axgrad/autograd/csrc)
target_link_libraries(tensor PRIVATE Python::Module)

if(WIN32)
  set_target_properties(tensor PROPERTIES SUFFIX ".pyd")
else()
  set_target_properties(tensor PROPERTIES PREFIX "lib")
endif()

install(TARGETS tensor DESTINATION axgrad COMPONENT python_modules)
install(DIRECTORY axgrad/ DESTINATION axgrad COMPONENT python_modules FILES_MATCHING PATTERN "*.py")