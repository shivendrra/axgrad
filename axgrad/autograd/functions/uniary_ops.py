from typing import Callable
from ...helpers.shape import *
from ...helpers.utils import _ones, _ones_like

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

class __UNSQUEEZE__:
  def __init__(self, first, out, dim) -> None: self.out, self.first, self.dim = out, first, dim
  def __call__(self) -> Callable:
    self.first.grad.data = unsqueeze(self.out.grad.data, self.dim)
    return self.__call__

class __SQUEEZE__:
  def __init__(self, first, out, dim) -> None: self.out, self.first, self.dim = out, first, dim
  def __call__(self) -> Callable:
    self.first.grad.data = squeeze(self.out.grad.data, self.dim)
    return self.__call__

class __FLATTEN__:
  def __init__(self, first, out, startdim=None, enddim=None) -> None: self.first, self.out, self.startdim, self.enddim = first, out, startdim, enddim
  def __call__(self) -> Callable:
    if self.startdim == None and self.enddim == None:
      self.first.grad.data = flatten(self.out.grad.data)
    else: self.first.grad.data = flatten_recursive(self.out.grad.data, self.startdim, self.enddim)
    return self.__call__

class __SWAPAXES__:
  def __init__(self, first, out, axis1, axis2) -> None: self.first, self.out, self.axis1, self.axis2 = first, out, axis1, axis2
  def __call__(self) -> Callable:
    self.first.grad.data = swap_axes(self.out.grad.data, self.axis1, self.axis2)
    return self.__call__

class __VIEW__:
  def __init__(self, first, out, original_shape) -> None: self.first, self.out, self.original_shape = first, out, original_shape
  def __call__(self) -> Callable:
    reshaped_grad = reshape(self.out.grad.data, self.original_shape)
    self.first.grad.data = reshaped_grad
    return self.__call__

class __SUM__:
  def __init__(self, first, out, axis=None, keepdims=False) -> None: self.first, self.out, self.axis, self.keepdims = first, out, axis, keepdims
  def __call__(self) -> Callable:
    grad_shape = get_shape(self.first.data)  
    if self.axis is None:  # sum over all elements
      grad_broadcast = _ones_like(self.first.data)
    else:
      if not self.keepdims:
        new_shape = list(grad_shape)
        new_shape[self.axis] = 1
        grad_broadcast = _ones(new_shape)
      else:
        grad_broadcast = _ones_like(self.out.grad.data)
    expanded_grad = flatten(broadcast(grad_broadcast, grad_shape))
    def mul_grad(flatten, grad):
      return [grad[0] * item for item in flatten]
    self.first.grad.data = reshape(mul_grad(expanded_grad, self.out.grad.data), tuple(grad_shape))
    return self.__call__

class __BROADCAST__:
  def __init__(self, first, out, new_shape) -> None: self.first, self.out, self.new_shape = first, out, new_shape
  def __call__(self) -> Callable:
    self.first.grad.data = broadcast(self.out.grad.data, self.new_shape)
    return self.__call__