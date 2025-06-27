from ctypes import c_float, c_size_t, c_int
from typing import *

from ._core import CTensor, lib, DType
from ._helpers import HDtype, HShape

int8, int16, int32, int64, long = "int8", "int16", "int32", "int64", "long"
float32, float64, double = "float32", "float64", "double"
uint8, uint16, uint32, uint64 = "uint8", "uint16", "uint32", "uint64"
boolean = "bool"

class Tensor:
  int8, int16, int32, int64, long, float32, float64, double, uint8, uint16, uint32, uint64, boolean = int8, int16, int32, int64, long, float32, float64, double, uint8, uint16, uint32, uint64, boolean
  
  def __init__(self, data: Union[List[Any], int, float], dtype: str=float32, requires_grad: bool=False):
    if isinstance(data, CTensor): self.data, self.shape, self.size, self.ndim, self.strides, self.dtype = data, (), 0, 0, [], dtype or "float32"
    elif isinstance(data, Tensor): self.data, self.shape, self.dtype, self.size, self.ndim, self.strides = data.data, data.shape, dtype or data.dtype, data.size, data.ndim, data.strides
    else:
      data, shape = HShape.flatten([data] if isinstance(data, (int, float)) else data), tuple(HShape.get_shape(data))
      self.size, self.ndim, self.dtype, self.shape, self.strides = len(data), len(shape), dtype or "float32", shape, HShape.get_strides(shape)
      self._data_ctypes, self._shape_ctypes = (c_float * self.size)(*data.copy()), (c_int * self.ndim)(*shape)
      self.data = lib.create_tensor(self._data_ctypes, c_size_t(self.ndim), self._shape_ctypes, c_size_t(self.size), c_int(HDtype._parse_dtype(self.dtype)))
    self.requires_grad, self.hooks, self.grad_fn, self.grad = requires_grad, [], None, None

  def __setattr__(self, name, value):
    if name == "grad": [setattr(self, "_temp_value", hook(getattr(self, "_temp_value", value))) for hook in self.hooks]; value = getattr(self, "_temp_value", value)
    super().__setattr__(name, value)
  def register_hook(self, function): self.hooks.append(function)
  def __str__(self): return (lib.print_tensor(self.data), "")[1]
  def astype(self, dtype: DType) -> "Tensor":
    out = Tensor(lib.cast_tensor(self.data, c_int(HDtype._parse_dtype(dtype))).contents, requires_grad=self.requires_grad)
    out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
    return (setattr(out, "grad", self.grad), setattr(out, "hooks", self.hooks), setattr(out, "grad_fn", self.grad_fn), out)[3] if self.requires_grad else out
  def tolist(self) -> List[Any]:
    data_ptr = lib.out_data(self.data)
    data_array = [data_ptr[i] for i in range(self.size)]
    if self.ndim == 0: return data_array[0]
    elif self.ndim == 1: return data_array
    else: return HShape.reshape_list(data_array, self.shape)

  # def __add__(self, other) -> "Tensor":
  #   if isinstance(other, (int, float)):
  #     result_pointer = lib.add_scalar_tensor(self.data, c_float(other)).contents
  #   else:
  #     other = other if isinstance(other, (CTensor, Tensor)) else Tensor(other)
  #     result_pointer = lib.add_tensor(self.data, other.data).contents
  #   out = Tensor(result_pointer, self.dtype, self.requires_grad)
  #   out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
  #   return out

  def __add__(self, other) -> "Tensor":
    result_ptr = lib.add_scalar_tensor(self.data, c_float(other)).contents if isinstance(other, (int, float)) else lib.add_tensor(self.data, (other if isinstance(other, (CTensor, Tensor)) else Tensor(other, self.dtype)).data).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def __sub__(self, other) -> "Tensor":
    result_ptr = lib.sub_scalar_tensor(self.data, c_float(other)).contents if isinstance(other, (int, float)) else lib.sub_tensor(self.data, (other if isinstance(other, (CTensor, Tensor)) else Tensor(other, self.dtype)).data).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def __mul__(self, other) -> "Tensor":
    result_ptr = lib.mul_scalar_tensor(self.data, c_float(other)).contents if isinstance(other, (int, float)) else lib.mul_tensor(self.data, (other if isinstance(other, (CTensor, Tensor)) else Tensor(other, self.dtype)).data).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def __truediv__(self, other) -> "Tensor":
    result_ptr = lib.div_scalar_tensor(self.data, c_float(other)).contents if isinstance(other, (int, float)) else lib.div_tensor(self.data, (other if isinstance(other, (CTensor, Tensor)) else Tensor(other, self.dtype)).data).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def __neg__(self) -> "Tensor":
    result_pointer = lib.neg_tensor(self.data).contents
    out = Tensor(result_pointer, self.dtype, self.requires_grad)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def __radd__(self, other): return self + other
  def __rsub__(self, other): return self - other
  def __rmul__(self, other): return self * other
  def __rtruediv__(self, other): return (self / other) ** -1

  def __pow__(self, exp) -> "Tensor":
    if isinstance(exp, (int, float)): result_ptr = lib.pow_tensor(self.data, c_float(exp)).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def __rpow__(self, base) -> "Tensor":
    if isinstance(base, (int, float)): result_ptr = lib.pow_scalar(c_float(base), self.data).contents
    else: raise NotImplementedError("__rpow__ with array base not implemented yet")
    out = Tensor(result_ptr, self.dtype, self.requires_grad)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def log(self) -> "Tensor":
    result_pointer = lib.log_tensor(self.data).contents
    out = Tensor(result_pointer, self.dtype, self.requires_grad)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def exp(self) -> "Tensor":
    result_pointer = lib.exp_tensor(self.data).contents
    out = Tensor(result_pointer, self.dtype, self.requires_grad)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def abs(self) -> "Tensor":
    result_pointer = lib.abs_tensor(self.data).contents
    out = Tensor(result_pointer, self.dtype, self.requires_grad)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def sqrt(self) -> "Tensor":
    result_pointer = lib.sqrt_tensor(self.data).contents
    out = Tensor(result_pointer, self.dtype, self.requires_grad)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def __matmul__(self, other):
    other = other if isinstance(other, (CTensor, Tensor)) else Tensor(other, self.dtype)
    if self.ndim <= 2 and other.ndim <= 2:
      result_ptr = lib.matmul_tensor(self.data, other.data).contents
    elif self.ndim == 3 and other.ndim == 3 and self.shape[0] == other.shape[0]:
      result_ptr = lib.batch_matmul_tensor(self.data, other.data).contents
    else:
      result_ptr = lib.broadcasted_matmul_tensor(self.data, other.data).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad)
    shape, ndim, size = lib.out_shape(out.data), self.ndim, lib.out_size(out.data)
    out.shape, out.ndim, out.size = tuple([shape[i] for i in range(ndim)]), ndim, size
    out.strides = HShape.get_strides(out.shape)
    return out

  def dot(self, other):
    other = other if isinstance(other, (CTensor, Tensor)) else Tensor(other, self.dtype)
    if self.ndim <= 2 and other.ndim <= 2:
      result_ptr = lib.dot_tensor(self.data, other.data).contents
    elif self.ndim == 3 and other.ndim == 3 and self.shape[0] == other.shape[0]:
      result_ptr = lib.batch_dot_tensor(self.data, other.data).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad)
    shape, ndim, size = lib.out_shape(result_ptr), out.data.ndim, lib.out_size(result_ptr)
    out.shape, out.ndim, out.size = tuple([shape[i] for i in range(ndim)]), ndim, size
    out.strides = HShape.get_strides(out.shape)
    return out