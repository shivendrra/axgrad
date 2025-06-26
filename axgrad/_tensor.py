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
    if isinstance(data, CTensor): self.data, self.shape, self.size, self.ndim, self.strides, self.dtype = data, (), 0, 0, [], dtype or "float32"
    elif isinstance(data, tensor): self.data, self.shape, self.dtype, self.size, self.ndim, self.strides = data.data, data.shape, dtype or data.dtype, data.size, data.ndim, data.strides
    else:
      data, shape = flatten([data] if isinstance(data, (int, float)) else data), tuple(get_shape(data))
      self.size, self.ndim, self.dtype, self.shape, self.strides = len(data), len(shape), dtype or "float32", shape, get_strides(shape)
      self._data_ctypes, self._shape_ctypes = (c_float * self.size)(*data.copy()), (c_int * self.ndim)(*shape)
      self.data = lib.create_tensor(self._data_ctypes, c_size_t(self.ndim), self._shape_ctypes, c_size_t(self.size), c_int(_parse_dtype(self.dtype)))
    self.requires_grad, self.hooks, self.grad_fn, self.grad = requires_grad, [], None, None

  def __setattr__(self, name, value):
    if name == "grad": [setattr(self, '_temp_value', hook(getattr(self, '_temp_value', value))) for hook in self.hooks]; value = getattr(self, '_temp_value', value)
    super().__setattr__(name, value)
  def register_hook(self, function): self.hooks.append(function)
  def __str__(self): return (lib.print_tensor(self.data), "")[1]
  def astype(self, dtype: DType) -> "tensor":
    out = tensor(lib.cast_tensor(self.data, c_int(_parse_dtype(dtype))).contents, requires_grad=self.requires_grad)
    out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
    return (setattr(out, 'grad', self.grad), setattr(out, 'hooks', self.hooks), setattr(out, 'grad_fn', self.grad_fn), out)[3] if self.requires_grad else out

  # def __add__(self, other) -> "tensor":
  #   if isinstance(other, (int, float)):
  #     result_pointer = lib.add_scalar_tensor(self.data, c_float(other)).contents
  #   else:
  #     other = other if isinstance(other, (CTensor, tensor)) else tensor(other)
  #     result_pointer = lib.add_tensor(self.data, other.data).contents
  #   out = tensor(result_pointer, self.dtype, self.requires_grad)
  #   out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
  #   return out

  def __add__(self, other) -> "tensor":
    result_ptr = lib.add_scalar_tensor(self.data, c_float(other)).contents if isinstance(other, (int, float)) else lib.add_tensor(self.data, (other if isinstance(other, (CTensor, tensor)) else tensor(other)).data).contents
    out = tensor(result_ptr, self.dtype, self.requires_grad)
    return (setattr(out, 'shape', self.shape), setattr(out, 'size', self.size), setattr(out, 'ndim', self.ndim), setattr(out, 'strides', self.strides), out)[4]

  def __sub__(self, other) -> "tensor":
    result_ptr = lib.sub_scalar_tensor(self.data, c_float(other)).contents if isinstance(other, (int, float)) else lib.sub_tensor(self.data, (other if isinstance(other, (CTensor, tensor)) else tensor(other)).data).contents
    out = tensor(result_ptr, self.dtype, self.requires_grad)
    return (setattr(out, 'shape', self.shape), setattr(out, 'size', self.size), setattr(out, 'ndim', self.ndim), setattr(out, 'strides', self.strides), out)[4]

  def __mul__(self, other) -> "tensor":
    result_ptr = lib.mul_scalar_tensor(self.data, c_float(other)).contents if isinstance(other, (int, float)) else lib.mul_tensor(self.data, (other if isinstance(other, (CTensor, tensor)) else tensor(other)).data).contents
    out = tensor(result_ptr, self.dtype, self.requires_grad)
    return (setattr(out, 'shape', self.shape), setattr(out, 'size', self.size), setattr(out, 'ndim', self.ndim), setattr(out, 'strides', self.strides), out)[4]

  def __truediv__(self, other) -> "tensor":
    result_ptr = lib.div_scalar_tensor(self.data, c_float(other)).contents if isinstance(other, (int, float)) else lib.div_tensor(self.data, (other if isinstance(other, (CTensor, tensor)) else tensor(other)).data).contents
    out = tensor(result_ptr, self.dtype, self.requires_grad)
    return (setattr(out, 'shape', self.shape), setattr(out, 'size', self.size), setattr(out, 'ndim', self.ndim), setattr(out, 'strides', self.strides), out)[4]

  def __neg__(self) -> "tensor":
    result_pointer = lib.neg_tensor(self.data).contents
    out = tensor(result_pointer, self.dtype, self.requires_grad)
    return (setattr(out, 'shape', self.shape), setattr(out, 'size', self.size), setattr(out, 'ndim', self.ndim), setattr(out, 'strides', self.strides), out)[4]

  def __radd__(self, other): return self + other
  def __rsub__(self, other): return self - other
  def __rmul__(self, other): return self * other
  def __rtruediv__(self, other): return (self / other) ** -1