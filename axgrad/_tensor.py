"""
  @tensor.py Main tensor class
  @breif Code contains axgrad.tensor class to perform backprop
  @comments
  - conjusted to save total lines of code
  - has basic functions & operations with backward function in same file
  - entrypoint to whole axgrad.tensor class & functions
"""

from typing import *
from copy import deepcopy
import math

from .helpers.shape import *
from ._dtype import *
from .helpers.utils import _zeros
from .helpers.ops import *
from .utils.contiguous import ContiguousOps

int8, int16, int32, int64, long = "int8", "int16", "int32", "int64", "long"
float16, float32, float64, double = "float16", "float32", "float64", "double"

class tensor:
  int8, int16, int32, int64, long, float16, float32, float64, double = "int8", "int16", "int32", "int64", "long", "float16", "float32", "float64", "double"
  def __init__(self, data, requires_grad:bool=True, dtype:Optional[Literal["int8", "int16", "int32", "int64", "float16", "float32", "float64", "long", "double"]]=None) -> None:
    if data is not None and isinstance(data, list):
      data = list(data)
    self.dtype = tensor.float32 if dtype is None else dtype
    self.data = Dtype.handle_conversion(data, self.dtype)
    self.shape, self.prev = self.shape(), set()
    self._backward = lambda: None
    self.stride = self.compute_stride(self.shape) # computing strides
    self.contiguous_ops = ContiguousOps(self) # creating an instance of coniguousops that works with this tensor
    # only if requires_grad is true
    if requires_grad is True:
      self.requires_grad, self.grad, self.grad_fn = requires_grad, _zeros(self.size), "<NotSet>"
    else:
      self.grad, self.grad_fn, self.requires_grad = None, None, False

  def __repr__(self) -> str:
    # basic representation for easy operation computing & no element or floating point truncation
    return f"tensor([{self.data}])"

  def __str__(self) -> str:
    # formated version computing for prettier outputs
    # truncates some of the elements & sub tensors if more than 8
    # displays additional information

    def format_element(element):
      if isinstance(element, list):
        return [format_element(sub_element) for sub_element in element]
      if self.dtype == int8 or self.dtype == int16 or self.dtype == int32 or self.dtype == int64 or self.dtype == long:
        return f"{element:.0f}."
      if self.dtype == float16:
        return f"{element:.2f}"
      if self.dtype == float32:
        return f"{element:.3f}"
      return f"{element:.4f}"

    formatted_data = format_element(self.data)

    def truncate_list(data, max_items=8):
      if len(data) > max_items:
        return data[:max_items // 2] + ['...'] + data[-max_items // 2:]
      return data

    def format_data(data, level=0):
      if isinstance(data[0], list):
        if len(data) > 8:
          data = truncate_list(data)  # Truncate rows if there are more than 8 arrays
        inner = ",\n".join(["  " * (level + 1) + format_data(sub_data, level + 1) for sub_data in data])
        return f"[\n{inner}\n" + "  " * level + "]"
      else:
        # Truncate individual row elements if they exceed 8
        data = truncate_list(data)
        return "[" + ", ".join(data) + "]"

    formatted_str = format_data(formatted_data, 0)
    formatted_str = formatted_str.replace("\t", " ")
    return f"tensor({formatted_str}, grad_fn={self.grad_fn})\n"  if self.requires_grad else f"tensor({formatted_str}, dtype={self.dtype})\n"
  
  # basic set & get item functions for tensor, not tested properly yet

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

  def __iter__(self) -> Iterator:
    for item in self.data:
      yield item
  
  # property attributes --------

  def shape(self) -> list:
    return get_shape(self.data)

  @property
  def size(self) -> tuple:
    return tuple(get_shape(self.data))
  
  @property
  def ndim(self) -> int:
    return len(self.size)
  
  @property
  def numel(self) -> int:
    out = 1
    out *= [dim for dim in self.size]
    return out
  
  @property
  def T(self) -> List["tensor"]:
    out = tensor(transpose(self.data), self.requires_grad, self.dtype)
    out.prev, out.grad, out.grad_fn = set(self, ), transpose(self.grad), "<TransposeBackwards>"
    return out
  
  @property
  def F(self) -> List["tensor"]:
    out = tensor(flatten(self.data), self.requires_grad, self.dtype)
    out.prev, out.grad, out.grad_fn = set(self, ), flatten(self.grad), "<FlattenBackwards>"
    return out

  def is_contiguous(self) -> bool:
    return self.contiguous_ops.is_contiguous()
  
  def make_contiguous(self) -> None:
    self.contiguous_ops.make_contiguous()
  
  def compute_stride(self, shape: List[int]) -> List[int]:
    return self.contiguous_ops.compute_stride(shape)
  
  def tolist(self) -> list:
    # returns the data into a list
    return list(self.data)
  
  def copy(self) -> List["tensor"]:
    out = tensor(deepcopy(self.data), self.requires_grad, self.dtype)
    out.prev, out.grad, out.grad_fn = self.prev, self.grad, self.grad_fn
    return out

  def astype(self, dtype:Optional[Literal["int8", "int16", "int32", "int64", "float16", "float32", "float64"]]) -> List["tensor"]:
    new_data = Dtype.handle_conversion(self.data, dtype)
    out = tensor(new_data, self.requires_grad, self.dtype)
    out.prev, out.grad, out.grad_fn = self.prev, self.grad, self.grad_fn
    return out

  def contiguous(self) -> List["tensor"]:
    # if the tensor is already flat or non-nested, return it as is
    if isinstance(self.data, list):
      reshaped_data = reshape(self.data, self.shape)
      self.data = reshaped_data
    return self

  def view(self, *new_shape:Union[int, list, tuple]) -> List["tensor"]:
    if isinstance(new_shape[0], list) or isinstance(new_shape[0], tuple):
      new_shape = tuple(new_shape[0])
    elif isinstance(new_shape[0], int):
      new_shape = tuple(new_shape)
    self.contiguous()
    flat_data = self.flatten()
    total_elements = len(flat_data)
    if total_elements != self.numel:
      raise ValueError("Total elements in new shape must match the number of elements in the original tensor")
    out = tensor(reshape(self.data, new_shape), requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev, out.grad_fn = set(self, ), "<ViewBackwards>"
    return out

  def reshape(self, *new_shape:Union[int, list, tuple]) -> List["tensor"]:
    if isinstance(new_shape[0], list) or isinstance(new_shape[0], tuple):
      new_shape = tuple(new_shape[0])
    elif isinstance(new_shape[0], int):
      new_shape = tuple(new_shape)
    
    out = tensor(reshape(self.data, new_shape), self.requires_grad, self.dtype)
    out.prev, out.grad_fn = set(self, ),  "<ReshapeBackwards>"
    return out
  
  def transpose(self) -> List["tensor"]:
    out = tensor(transpose(self.data), self.requires_grad, self.dtype)
    out.prev, out.grad_fn = set(self, ), "<TransposeBackwards>"
    return out
  
  def swapaxes(self, dim0:int, dim1:int) -> List["tensor"]:
    out = tensor(swap_axes(self.data, dim0, dim1), self.requires_grad, self.dtype)
    out.prev, out.grad_fn = set(self, ), "<TransposeBackwards>"
    return out
  
  def flatten(self, start_dim:int, end_dim:int) -> List["tensor"]:
    out = tensor(flatten_recursive(self.data, start_dim, end_dim), self.requires_grad, self.dtype)
    out.prev, out.grad_fn = set(self, ), "<FlattenBackwards>"
    return out
  
  def unsqueeze(self, dim:int=0):
    dim = dim if dim > 0 else self.ndim + dim
    out = tensor(unsqueeze(self.data, dim), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev, out.grad_fn = (self, ), "<UnsqueezeBackwards>"
    return out
  
  def squeeze(self, dim:int=0):
    if dim is not None and dim>=self.ndim:
      raise IndexError(f"Dimension out of range (expected to be in range of {self.ndim} dimensions)")
    dim = dim if dim > 0 else self.ndim + dim
    out = tensor(squeeze(self.data, dim), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev, out.grad_fn = (self, ), "<SqueezeBackwards>"
    return out
  
  def sum(self, axis:Optional[int]=None, keepdims:bool=False):
    if axis == None:
      if keepdims:
        out = [[sum(flatten(self.data))]]
      else:
        out = sum(flatten(self.data))
    elif axis == 0:
      out = sum_axis0(self.data)
    else:
      out = sum_axis(self.data, axis, keepdims)
    out = tensor(out, self.requires_grad, self.dtype)
    out.prev, out.grad_fn = set(self, ), "<SumBackwards>"
    return out
  
  def dot(self, other:List["tensor"]) -> List["tensor"]:
    out = tensor(dot_product(self.data, other.data), self.requires_grad, self.dtype)
    out.prev, out.grad_fn = set(self.data), "<DotBackwards>"
    return out
  
  def det(self) -> List["tensor"]:
    out = tensor(determinant(self.data), self.requires_grad, self.dtype)
    out.prev, out.grad_fn = set(self, ), "<DetBackwards>"
    return out

  def mean(self, axis:Optional[int]=None, keepdims:bool=False) -> List["tensor"]:
    if axis is None:
      out = sum(flatten(self.data)) / self.numel
      if keepdims:
        out = [[out]]
    elif axis == 0:
      out = mean_axis0(self.data)
    else:
      out = mean_axis(self.data, axis, keepdims)
    out = tensor(out, self.requires_grad, self.dtype)
    out.prev, out.grad_fn = set(self, ), "<MeanBackwards>"
    return out
  
  def var(self, axis:Optional[int]=None, ddof:int=0, keepdims:bool=False) -> List["tensor"]:
    if axis is None:
      flat_array = flatten(self.data)
      mean_val = sum(flat_array) / self.numel
      out = sum((x - mean_val) ** 2 for x in flat_array) / (len(flat_array) - ddof)
      if keepdims:
        out = [[out]]
      return out
    elif axis == 0:
      out = var_axis0(self.data)
    else:
      mean_val = self.mean(axis=axis).data
      out = var_axis(self.data, mean_val, axis, ddof, keepdims)
    out = tensor(out, self.requires_grad, self.dtype)
    out.prev, out.grad_fn = set(self, ), "<VarBackwards>"
    return out
  
  def std(self, axis:Optional[int]=None, ddof:int=0, keepdims:bool=False) -> List["tensor"]:
    variance = self.var(axis, ddof, keepdims).data
    def _std(var):
      if isinstance(var, list):
        return [_std(sub) for sub in var]
      return math.sqrt(var)
    if keepdims:
      out = [[math.sqrt(x)] for x in flatten(variance)]
    else:
      out = _std(variance)
    out = tensor(out, self.requires_grad, self.dtype)
    out.prev, out.requires_grad = set(self, ), "<StdBackwards>"
    return out

  def __add__(self, other) -> List["tensor"]:
    other = other if isinstance(other, tensor) else other = tensor(other, requires_grad=self.requires_grad, dtype=self.dtype)
    def _ops(a, b):
      if isinstance(a, list):
        return [_ops(_a, _b) for _a, _b in zip(a, b)]
      else:
        return a + b
    target_shape, requires_broadcasting = broadcast_shape(self.shape, other.shape)
    if requires_broadcasting:
      self.data, self.grad, self.shape = Dtype.handle_conversion(broadcast(self.data, target_shape), self.dtype), broadcast(self.grad, target_shape), get_shape(self.data)
      other.data, other.grad, other.shape = Dtype.handle_conversion(broadcast(other.data, target_shape), other.dtype), broadcast(other.grad, target_shape), get_shape(other.data)
    if self.size == other.size:
      out = tensor(_ops(self.data, other.data), dtype=self.dtype, requires_grad=self.requires_grad)
      out.prev, out.grad_fn = (self, other), "<AddBackwards>"
    else:
      raise ValueError("shapes are incompatible for operation")
    return out
  
  def __mul__(self, other) -> List["tensor"]:
    other = other if isinstance(other, tensor) else other = tensor(other, requires_grad=self.requires_grad, dtype=self.dtype)
    def _ops(a, b):
      if isinstance(a, list):
        return [_ops(_a, _b) for _a, _b in zip(a, b)]
      else:
        return a * b
    target_shape, requires_broadcasting = broadcast_shape(self.shape, other.shape)
    if requires_broadcasting:
      self.data, self.grad, self.shape = Dtype.handle_conversion(broadcast(self.data, target_shape), self.dtype), broadcast(self.grad, target_shape), get_shape(self.data)
      other.data, other.grad, other.shape = Dtype.handle_conversion(broadcast(other.data, target_shape), other.dtype), broadcast(other.grad, target_shape), get_shape(other.data)
    if self.size == other.size:
      out = tensor(_ops(self.data, other.data), dtype=self.dtype, requires_grad=self.requires_grad)
      out.prev, out.grad_fn = (self, other), "<MulBackwards>"
    else:
      raise ValueError("shapes are incompatible for operation")
    return out
  
  def __matmul__(self, other) -> List["tensor"]:
    other = other if isinstance(other, tensor) else other = tensor(other, requires_grad=self.requires_grad, dtype=self.dtype)

    if self.size == other.size:
      out, self.grad, other.grad = matmul(self.data, other.data, self.grad, other.grad)
      out = tensor(out, dtype=self.dtype, requires_grad=self.requires_grad)
      out.prev, out.grad_fn = (self, other), "<MatmulBackwards>"
    else:
      raise ValueError("shapes are incompatible for operation")
    return out
  
  def __neg__(self) -> List["tensor"]:
    def _ops(data):
      if isinstance(data, list):
        return [_ops(d) for d in data]
      return -data
    out = tensor(_ops(self.data), self.requires_grad, self.dtype)
    out.prev, out.grad_fn = set(self, ), "<NegBackwards>"

    def neg_backward():
      def _neg(grad, out):
        if not isinstance(grad, list):
          return -out
        return [_neg(g, og) for g, og, in zip(grad, out)]
      self.grad = _neg(self.grad, out.data)
    out._backward = neg_backward
    return out

  def __radd__(self, other) -> List["tensor"]:
    other = other if isinstance(other, tensor) else tensor(other, self.requires_grad, self.dtype)
    return other + self

  def __rmul__(self, other) -> List["tensor"]:
    other = other if isinstance(other, tensor) else tensor(other, self.requires_grad, self.dtype)
    return other * self
  
  def __sub__(self, other) -> List["tensor"]:
    other = other if isinstance(other, tensor) else tensor(other, self.requires_grad, self.dtype)
    return self + (-other)
  
  def __rsub__(self, other) -> List["tensor"]:
    other = other if isinstance(other, tensor) else tensor(other, self.requires_grad, self.dtype)
    return other + (-self)
  
  def __pow__(self, pow:Union[int, float], eps:float=1e6) -> List["tensor"]:
    assert isinstance(pow, (int, float)), "power exponent is of incompatible datatype"
    other = other if isinstance(other, tensor) else tensor(other, self.requires_grad, self.dtype)

    def _ops(data, pow):
      if isinstance(data, list):
        return [_ops(_d, pow) for _d in data]
      if data == 0:
        data = eps
      return math.pow(data, pow)
    out = tensor(_ops(self.data), self.requires_grad, self.dtype)
    out.prev, out.grad_fn = set(self, ), "<PowBackwards>"
    return out
  
  def __truediv__(self, other) -> List["tensor"]:
    other = other if isinstance(other, tensor) else tensor(other, self.requires_grad, self.dtype)
    return self + (other ** -1)
  
  def __rtruediv__(self, other) -> List["tensor"]:
    other = other if isinstance(other, tensor) else tensor(other, self.requires_grad, self.dtype)
    return other + (self ** -1)