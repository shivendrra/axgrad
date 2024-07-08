import math
from copy import deepcopy
from typing import *

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
    self.ndim = len(self.shape)
    self.dtype = tensor.int64 if dtype is None else dtype
    self.T = self.transpose()
    self.requires_grad = requires_grad
    self.grad = 0 if self.requires_grad else None
    self.prev = tuple(child) if child is not None else ()
    self._backward = lambda x: None
    self.ops = ops
    if dtype is not None:
      self.dtype = self._convert_dtype(self.data, dtype)
  
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