"""
  @core.py
  @brief: main multi-dim tensor class to build tensor manipulation functions & ops
  * tensor() & grad() is going to be built over on top of this, for proper grad management
  * no backprop enabled
"""

from typing import *
from copy import deepcopy

from ._dtype import Dtype
from .utils.contiguous import ContiguousOps
from .helpers.shape import get_shape, get_strides, get_size, transpose, flatten_recursive, unsqueeze, squeeze, swap_axes, reshape
from .helpers.ops import broadcast, dot_product, determinant
from .ops.main import *
from .ops.binary import register_binary_operators
from .ops.reduction import register_reduction_operators

int8, int16, int32, int64, long = "int8", "int16", "int32", "int64", "long"
float16, float32, float64, double = "float16", "float32", "float64", "double"

class _tensor:
  int8, int16, int32, int64, long, float16, float32, float64, double = int8, int16, int32, int64, long, float16, float32, float64, double
  def __init__(self, data, dtype:Optional[Literal["int8", "int16", "int32", "int64", "float16", "float32", "float64", "long", "double"]]=None) -> None:
    if data is not None and isinstance(data, list):
      data = list(data)
    self.dtype = _tensor.float32 if dtype is None else dtype
    self.shape = get_shape(data)
    self.data = Dtype.handle_conversion(data, self.dtype)
    self.size = get_size(self.shape)
    self.contiguous_ops = ContiguousOps(self) # creating an instance of coniguousops that works with this tensor
    self.stride = get_strides(self.shape) # computing strides
    self.is_scalar = True if self.size == (1) or self.size == (1,1) or self.size == (1,) else False

  def __getitem__(self, index:tuple):
    if isinstance(index, tuple):
      data = self.data
      for idx in index[:-1]:
        data = data[idx]
      return data[index[-1]]
    else:
      return self.data[index]

  def __setattr__(self, name: str, value: Any) -> None:
    super().__setattr__(name, value)
  
  def __setitem__(self, index:tuple, value: Any) -> None:
    if isinstance(index, tuple):
      data = self.data
      for idx in index[:-1]:
        data = data[idx]
      data[index[-1]] = value
    else:
      self.data[index] = value

  def __iter__(self) -> Iterator: return (item for item in self.data)
  def __repr__(self) -> str: return f"{self.data}"
  def __len__(self) -> int: return self.size
  def is_contiguous(self) -> bool: return self.contiguous_ops.is_contiguous()
  def make_contiguous(self) -> None: self.contiguous_ops.make_contiguous()
  def compute_stride(self, shape: List[int]) -> List[int]: return self.contiguous_ops.compute_stride(shape)
  def tolist(self) -> list: return list(self.data)
  def copy(self) -> "_tensor": return _tensor(deepcopy(self.data), self.dtype)
  def transpose(self) -> "_tensor": return _tensor(transpose(self.data), self.dtype)
  def flatten(self, start_dim: int, end_dim: int) -> "_tensor": return _tensor(flatten_recursive(self.data, start_dim, end_dim), self.dtype)
  def unsqueeze(self, dim: int=0) -> "_tensor": return _tensor(unsqueeze(self.data, dim), self.dtype)
  def sequeeze(self, dim: int=0) -> "_tensor": return _tensor(squeeze(self.data, dim), self.dtype)
  def reshape(self, new_shape: tuple) -> "_tensor": return _tensor(reshape(self.data, new_shape), self.dtype)
  def __pow__(self, other) -> "_tensor": return _tensor(pow_tensor(self.data, other), self.dtype)
  def dot(self, other: "_tensor") -> "_tensor": return _tensor(dot_product(self.data, other.data), self.dtype)
  def det(self) -> "_tensor": return _tensor(determinant(self.data), self.dtype)
  def exp(self) -> "_tensor": return _tensor(exp_tensor(self.data), self.dtype)
  def log(self) -> "_tensor": return _tensor(log_tensor(self.data), self.dtype)
  def ln(self) -> "_tensor": return _tensor(ln_tensor(self.data), self.dtype)
  def sin(self) -> "_tensor": return _tensor(sin_tensor(self.data), self.dtype)
  def sinh(self) -> "_tensor": return _tensor(sinh_tensor(self.data), self.dtype)
  def cos(self) -> "_tensor": return _tensor(cos_tensor(self.data), self.dtype)
  def cosh(self) -> "_tensor": return _tensor(cosh_tensor(self.data), self.dtype)
  def tan(self) -> "_tensor": return _tensor(tan_tensor(self.data), self.dtype)
  def tanh(self) -> "_tensor": return _tensor(tanh_tensor(self.data), self.dtype)
  def relu(self) -> "_tensor": return _tensor(relu_tensor(self.data), self.dtype)
  def gelu(self) -> "_tensor": return _tensor(gelu_tensor(self.data), self.dtype)
  def sigmoid(self) -> "_tensor": return _tensor(sigmoid_tensor(self.data), self.dtype)
  def leaky_relu(self) -> "_tensor": return _tensor(leaky_relu_tensor(self.data), self.dtype)

  def view(self, *new_shape:Union[int, list, tuple]):
    if isinstance(new_shape[0], list) or isinstance(new_shape[0], tuple):
      new_shape = tuple(new_shape[0])
    elif isinstance(new_shape[0], int):
      new_shape = tuple(new_shape)
    self.make_contiguous()
    flat_data = self.flatten()
    total_elements = len(flat_data)
    if total_elements != self.size:
      raise ValueError("Total elements in new shape must match the number of elements in the original tensor")
    out = _tensor(reshape(self.data, new_shape), dtype=self.dtype)
    return out

register_binary_operators()
register_reduction_operators()