import math
from copy import deepcopy
from typing import *
from .helpers.shapes import *
from .dtype.convert import *

int8 = 'int8'
int16 = 'int16'
int32 = 'int32'
int64 = 'int64'
float16 = 'float16'
float32 = 'float32'
float64 = 'float64'
long = 'long'
double = 'double'

class tensor:
  int8 = int8
  int16 = int16
  int32 = int32
  int64 = int64
  long = long
  float16 = float16
  float32 = float32
  float64 = float64
  double = double

  def __init__(self,
              *data: Union[List['tensor'], list, int, float],
              dtype: Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None,
              requires_grad: bool=False,
              child: set=None,
              ops: str=None
              ) -> None:
    self.data = data[0] if len(data) == 1 and isinstance(data[0], list) else list(data)
    self.shape = self.shape()
    self.size = tuple(get_shape(self.data))
    self.ndim = len(self.shape)
    self.dtype = tensor.int64 if dtype is None else dtype
    self.T = self.transpose()
    self.requires_grad = requires_grad
    self.grad = 0 if self.requires_grad else None
    self.prev = tuple(child) if child is not None else ()
    self._backward = lambda x: None
    self.ops = ops
    self.grad_fn = None
    self.T = self.transpose()
    if dtype is not None:
      self.dtype = convert_dtype(self.data, dtype)
  
  def __repr__(self) -> str:
    if self.ndim == 1:
      return f"array([{self.data}], dtype={self.dtype})"
    data_str = ',\n\t'.join([str(row) for row in self.data])
    return f"array([{data_str}], dtype={self.dtype})"
  
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
  
  def shape(self) -> list:
    return get_shape(self.data)
  
  def numel(self):
    out = 1
    for dim in self.shape:
      out *= dim
    return out

  def swap_axes(self, dim0, dim1, depth=0) -> List['tensor']:
    out = swap_axes(self.data, dim0, dim1, self.ndim, depth)
    return out
  
  def transpose(self) -> list:
    out = transpose(self.data)
    return tensor(out, child=(self, ), dtype=self.dtype, ops='<transpose>', requires_grad=self.requires_grad)
  
  def broadcast(self, shape:Union[tuple, list]) -> List['tensor']:
    out = broadcast(self.data, shape)
    return tensor(out, child=(self, ), dtype=self.dtype, ops='<broadcast>', requires_grad=self.requires_grad)

  def detach(self) -> None:
    self.grad = None
    self.grad_fn = None
  
  def zero_grad(self) -> None:
    self.grad = None
  
  def copy(self) -> List['tensor']:
    out = tensor(deepcopy(self.data), child=tuple(self.prev), dtype=self.dtype, requires_grad=self.requires_grad, ops='<copy>')
    return out
  
  def tolist(self) -> list:
    return self.data
  
  def flatten(self) -> List['tensor']:
    out = tensor(flatten(self.data), child=(self,), dtype=self.dtype, ops='<flatten>', requires_grad=self.requires_grad)
    return out
  
  def unsqueeze(self, dim:int=0) -> List['tensor']:
    out = tensor(unsqueeze(self.data, dim), child=(self,), dtype=self.dtype, requires_grad=self.requires_grad, ops='<unsqueeze>')
    return out
  
  def reshape(self, new_shape:tuple) -> List['tensor']:
    new_shape = new_shape if isinstance(new_shape, tuple) else tuple(new_shape)
    out = tensor(reshape(self.data, new_shape), child=(self,), dtype=self.dtype, requires_grad=self.requires_grad, ops='<resphape>')
    return out
  
  def squeeze(self, dim:int=None) -> List['tensor']:
    if dim is not None and (dim<0 or dim>self.ndim):
      raise IndexError(f"Dimensions out of range (expected to be in range of {self.ndim} dimensions)")

    out = tensor(squeeze(self.data, dim), child=(self, ), dtype=self.dtype, requires_grad=self.requires_grad, ops='<squeeze>')
    return out
  
  def sum(self, axis=None, keepdim=False):
    pass