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
  
  def is_contiguous(self) -> List["tensor"]:
    pass

  def contiguous(self) -> List["tensor"]:
    pass

  def view(self, *new_shape:Union[int, list, tuple]) -> List["tensor"]:
    if isinstance(new_shape[0], list) or isinstance(new_shape[0], tuple):
      new_shape = tuple(new_shape[0])
    elif isinstance(new_shape[0], int):
      new_shape = tuple(new_shape)
    pass

  def reshape(self, *new_shape:Union[int, list, tuple]) -> List["tensor"]:
    if isinstance(new_shape[0], list) or isinstance(new_shape[0], tuple):
      new_shape = tuple(new_shape[0])
    elif isinstance(new_shape[0], int):
      new_shape = tuple(new_shape)
    
    out = tensor(reshape(self.data, new_shape), self.requires_grad, self.dtype)
    out.prev, out.grad, out.grad_fn = set(self, ), reshape(self.grad), "<ReshapeBackwards>"
    return out