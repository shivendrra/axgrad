from typing import Any
from helpers.shape import get_shape, _flatten
import math

class tensor:
  def __init__(self, *data, child=(), requires_grad=True):
    self.data = data[0] if len(data) == 1 and isinstance(data[0], list) else list(data)
    self.shape = self.shape()
    self.ndim = len(self.shape)
    self.requires_grad = requires_grad
    self.grad = 0 if self.requires_grad else None
    self.prev = set(child)
    self.leaf = set()
    self._backward = lambda: None
    self.grad_fn = None
  
  def __repr__(self) -> str:
    return f"tensor({self.data})"
  
  def __getitem__(self, index):
    return self.data[index]
  
  def __setattr__(self, name: str, value: Any) -> None:
    pass

  def __setitem__(self, index, value):
    if isinstance(index, tuple):
      data = self.data
      grad = self.grad
      for idx in index[:-1]:
        data = data[idx]
        grad = grad[idx]
      data[index[-1]] = value
      grad[index[-1]] = value
    else:
      self.data[index] = value
      self.grad[index] = value
  
  def shape(self):
    return get_shape(self.data)
  
  def flatten(self):
    return _flatten(self.data)
  
  def __add__(self, other):
    other = other if isinstance(other, tensor) else tensor(other)