from typing import Any, Callable
from ...helpers.shape import *

class __TRANSPOSE__:
  def __init__(self, first, out) -> None: self.first, self.out = first, out
  def __call__(self) -> Callable:
    self.first.grad.data = transpose(self.out.grad.data)
    return self.__call__

class __RESHAPE__:
  def __init__(self, first, out, new_shape) -> None: self.first, self.out, self.new_shape = first, out, new_shape
  def __call__(self) -> Callable:
    self.first.grad.data = reshape(self.out.grad.data, self.new_shape)
    return self.__call__

class __SWAPAXES__:
  def __init__(self, first, out, axis1, axis2) -> None: self.first, self.out, self.axis1, self.axis2 = first, out, axis1, axis2
  def __call__(self) -> Callable:
    self.first.grad.data = swap_axes(self.out.grad.data, self.axis1, self.axis2)
    return self.__call__