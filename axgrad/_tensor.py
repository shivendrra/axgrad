"""
  @tensor.py Main tensor class
  @breif Code contains axgrad.tensor class to perform backprop
  @comments
  - congested to save total lines of code
  - has basic functions & operations with backward function in same file
  - entrypoint to whole axgrad.tensor class & functions
"""

from ctypes import c_float, c_size_t, c_int
from typing import *
from copy import deepcopy

from ._core import CTensor, lib, DType
from ._helpers import get_shape, get_strides, flatten

int8, int16, int32, int64, long = "int8", "int16", "int32", "int64", "long"
float32, float64, double = "float32", "float64", "double"
uint8, uint16, uint32, uint64 = "uint8", "uint16", "uint32", "uint64"
boolean = "bool"

def _parse_dtype(dtype: str) -> int:
  dtype_map = {"float32": DType.FLOAT32, "float64": DType.FLOAT64, "int8": DType.INT8, "int16": DType.INT16, "int32": DType.INT32, "int64": DType.INT64, "uint8": DType.UINT8, "uint16": DType.UINT16, "uint32": DType.UINT32, "uint64": DType.UINT64, "bool": DType.BOOL,}
  if dtype not in dtype_map:
    raise ValueError(f"Unsupported dtype: {dtype}. Supported dtypes: {list(dtype_map.keys())}")
  return dtype_map[dtype]

class tensor:
  int8, int16, int32, int64, long, float32, float64, double, uint8, uint16, uint32, uint64, boolean = int8, int16, int32, int64, long, float32, float64, double, uint8, uint16, uint32, uint64, boolean
  def __init__(self, data: Union[List[Any], int, float], dtype: str=float32, requires_grad: bool=False):
    if isinstance(data, CTensor):
      self.data, self.shape = data, ()
      self.size, self.ndim, self.strides = 0, 0, []
      self.dtype = dtype if dtype else "float32"
    elif isinstance(data, tensor):
      self.data, self.shape = data.data, data.shape
      self.size, self.ndim, self.strides = data.size, data.ndim, data.strides
      self.dtype = dtype if dtype else data.dtype
    else:
      if isinstance(data, (int, float)):
        data = [data]
      data, shape = flatten(data), tuple(get_shape(data))
      self.size, self.ndim = len(data), len(shape)
      dtype = "float32" if dtype is None else dtype
      self.shape, self.dtype, self.strides = deepcopy(shape), dtype, get_strides(shape)

      # ctypes arrays
      self._data_ctypes = (c_float * self.size)(*data.copy())
      self._shape_ctypes = (c_int * self.ndim)(*shape)

      self.data = lib.create_tensor(self._data_ctypes, c_size_t(self.ndim), self._shape_ctypes, c_size_t(self.size), c_int(_parse_dtype(dtype)))
    self.requires_grad, self.hooks, self.grad_fn, self.grad = requires_grad, [], None, None

  def __setattr__(self, name, value):
    if name == "grad":
      for hook in self.hooks:
        value = hook(value)
    super().__setattr__(name, value)

  def register_hook(self, function):
    self.hooks.append(function)

  def __str__(self):
    lib.print_tensor(self.data)
    return ""