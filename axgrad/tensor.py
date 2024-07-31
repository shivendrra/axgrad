from typing import *
from .helpers.shape import *
from .dtypes.convert import handle_conversion
from .helpers.utils import *
from copy import deepcopy
from .helpers.functionals import *

int8 = 'int8'
int16 = 'int16'
int32 = 'int32'
int64 = 'int64'
long = 'long'
float16 = 'float16'
float32 = 'float32'
float64 = 'float64'
double = 'double'

class tensor:
  int8 = int8
  int16 = int16
  int32 = int32
  int64 = int64 or long
  float16 = float16
  float32 = float32
  float64 = float64 or double

  def __init__(self, *data, requires_grad:bool=True, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None) -> None:
    if len(data) == 1 and isinstance(data[0], list):
      self.data = data[0]
    else:
      list(data)
    self.dtype = tensor.int32 if dtype is None else dtype
    if dtype is not None:
      self.data = handle_conversion(self.data, dtype)
    self.shape = self.shape()
    self.size = tuple(self.shape)
    self.ndim = len(self.shape)
    self.prev = set()
    self.grad_fn = None
    self.grad = None
    self.requires_grad = requires_grad
    if self.requires_grad is True:
      self.grad = zeros(self.shape)
    self._backward = lambda: None
  
  def __repr__(self) -> str:
    if self.ndim == 1:
      return f'tensor([{self.data}], dtype={self.dtype})'
    data_str = ',\n\t'.join([str(row) for row in self.data])
    return f'tensor([{data_str}], dtype={self.dtype})'

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

  def astype(self, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]) -> List['tensor']:
    new_data = handle_conversion(self.data, dtype)
    return tensor(new_data, requires_grad=self.requires_grad)

  def tolist(self) -> list:
    return self.data
  
  def copy(self) -> List['tensor']:
    return tensor(deepcopy(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
  
  def shape(self) -> list:
    return get_shape(self.data)
  
  def flatten(self, start_dim:int=0, end_dim:int=-1) -> List['tensor']:
    out = tensor(flatten_recursive(self.data, start_dim, end_dim), requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev = set(self,)
    return out
  
  def F(self) -> List['tensor']:
    out = tensor(flatten(self.data), requires_grad=self.requires_grad, dtype=self.data)
    out.prev = set(self,)
  
  def transpose(self, dim) -> List['tensor']:
    out = tensor(transpose_recursive(self.data, dim), requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev = set(self,)
    return out
  
  def T(self) -> List['tensor']:
    out = tensor(transpose(self.data), requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev = set(self,)
    return out
  
  def reshape(self, new_shape:tuple) -> List['tensor']:
    out = tensor(reshape(self.data), requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev = set(self,)
    return out
  
  