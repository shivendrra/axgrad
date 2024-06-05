from typing import Any
from .helpers.shape import get_shape, _flatten, _unsqueeze
import math

class tensor:
  def __init__(self, *data, child=(), requires_grad=False):
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
    data_str = ',\n\t'.join([str(row) for row in self.data])
    return f'tensor({data_str})'
  
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

  def _infer_shape(self, data):
    if isinstance(data, list):
      return [len(data)] + self._infer_shape(data[0])
    return []

  def shape(self):
    return get_shape(self.data)

  def flatten(self, start_dim:int, end_dim:int):
    out = tensor(_flatten(self.data, start_dim, end_dim), child=(self,))
    return out

  def unsqueeze(self, dim:int=0):
    out = tensor(_unsqueeze(self.data, dim), child=(self,))
    return out

  def __add__(self, other):
    other = other if isinstance(other, tensor) else tensor(other)