from typing import Callable
from ...helpers.shape import *
from ...helpers.utils import _ones, _ones_like

def elementwise_mul(a, b):
  if isinstance(a, list) and isinstance(b, list):
    return [elementwise_mul(a_elem, b_elem) for a_elem, b_elem in zip(a, b)]
  elif isinstance(a, list) and isinstance(b, (int, float)):
    return [elementwise_mul(a_elem, b) for a_elem in a]
  elif isinstance(a, (int, float)) and isinstance(b, list):
    return [elementwise_mul(a, b_elem) for b_elem in b]
  else:
    return a * b

def elementwise_sub(a, b):
  if isinstance(a, list) and isinstance(b, list):
    return [elementwise_sub(a_elem, b_elem) for a_elem, b_elem in zip(a, b)]
  elif isinstance(a, list) and isinstance(b, (int, float)):
    return [elementwise_sub(a_elem, b) for a_elem in a]
  elif isinstance(a, (int, float)) and isinstance(b, list):
    return [elementwise_sub(a, b_elem) for b_elem in b]
  else:
    return a - b

def _div_scalar(tensor, scalar):
  if isinstance(tensor, list):
    return [_div_scalar(elem, scalar) for elem in tensor]
  else:
    return tensor / scalar

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

class __MEAN__:
  def __init__(self, first, out, axis=None, keepdims=False) -> None: self.first, self.out, self.axis, self.keepdims = first, out, axis, keepdims
  def __call__(self) -> Callable:
    grad_shape = get_shape(self.first.data)
    if self.axis is None:
      grad_broadcast = _ones_like(self.first.data)
      factor = self.first.numel
    else:
      if not self.keepdims:
        new_shape = list(grad_shape)
        new_shape[self.axis] = 1
        grad_broadcast = _ones(new_shape)
      else:
        grad_broadcast = _ones_like(self.out.grad.data)

      factor = grad_shape[self.axis]
    scaled_grad = _div_scalar(grad_broadcast, factor)
    expanded_grad = broadcast(scaled_grad, grad_shape)
    self.first.grad.data = elementwise_mul(expanded_grad, self.out.grad.data)
    return self.__call__

class __VAR__:
  def __init__(self, first, out, axis=None, ddof=0, keepdims=False) -> None: self.first, self.out, self.axis, self.ddof, self.keepdims = first, out, axis, 0, keepdims
  def __call__(self) -> Callable:
    N = self.first.numel if self.axis is None else get_shape(self.first.data)[self.axis]
    grad_shape = get_shape(self.first.data)

    if self.axis is None:
      grad_broadcast = _ones_like(self.first.data)
    else:
      if not self.keepdims:
        new_shape = list(grad_shape)
        new_shape[self.axis] = 1
        grad_broadcast = _ones(new_shape)
      else:
        grad_broadcast = _ones_like(self.out.grad.data)
    mean_val = self.first.mean(self.axis, keepdims=True).data
    mean_broadcast = broadcast(mean_val, grad_shape)

    scaled_grad = _div_scalar(grad_broadcast, N - self.ddof)
    expanded_grad = broadcast(scaled_grad, grad_shape)
    diff = elementwise_sub(self.first.data, mean_broadcast)
    grad_contrib = elementwise_mul(2 * diff, expanded_grad)
    self.first.grad.data = elementwise_mul(grad_contrib, self.out.grad.data)
    return self.__call__

class __STD__:
  def __init__(self, first, out, axis=None, ddof=0, keepdims=False) -> None: self.first, self.out, self.axis, self.ddof, self.keepdims = first, out, axis, 0, keepdims
  def __call__(self) -> Callable:
    N = self.first.numel if self.axis is None else get_shape(self.first.data)[self.axis]
    grad_shape = get_shape(self.first.data)

    if self.axis is None:
      grad_broadcast = _ones_like(self.first.data)
    else:
      if not self.keepdims:
        new_shape = list(grad_shape)
        new_shape[self.axis] = 1
        grad_broadcast = _ones(new_shape)
      else:
        grad_broadcast = _ones_like(self.out.grad.data)
    mean_val = self.first.mean(self.axis, keepdims=True).data
    mean_broadcast = broadcast(mean_val, grad_shape)
    scaled_grad = _div_scalar(grad_broadcast, N - self.ddof)
    expanded_grad = broadcast(scaled_grad, grad_shape)
    diff = elementwise_sub(self.first.data, mean_broadcast)
    grad_contrib = elementwise_mul(2 * diff, expanded_grad)

    # Chain rule: std = sqrt(var) => dstd/dx = (1 / (2 * std)) * dvar/dx
    std_val = self.out.data
    std_broadcast = broadcast(std_val, grad_shape)
    std_grad_contrib = _div_scalar(grad_contrib, 2 * std_broadcast)
    self.first.grad.data = elementwise_mul(std_grad_contrib, self.out.grad.data)
    
    return self.__call__

class __SQRT__:
  def __init__(self, first, out) -> None: self.first, self.out = first, out
  def backward(self, grad, out):
    if not isinstance(grad, list):
      # Gradient for sqrt: 0.5 * (x ** -0.5)
      grad += (0.5 * (out ** -1)) * out
      return grad
    return [self.backward(g, og) for g, og in zip(grad, out)]

  def __call__(self) -> Callable:
    self.first.grad.data = self.backward(self.first.grad.data, self.out.grad.data)
    return self.__call__

class __CLIP__:
  def __init__(self, first, out, _min, _max) -> None: self.first, self.out, self._max, self._min = first, out, _max, _min
  def backward(self, grad, input_data, _min, _max):
    if isinstance(grad, list):
      return [self.backward(g, idata, _min, _max) for g, idata in zip(grad, input_data)]
    return grad if _min <= input_data <= _max else 0.0

  def __call__(self) -> Callable:
    self.first.grad.data = self.backward(self.out.grad.data, self.out.data, self._min, self._max)
    return self.__call__

class __EXP__:
  def __init__(self, first, out) -> None: self.first, self.out = first, out
  def backward(self, grad, out):
    if not isinstance(grad, list):
      grad += out * out
      return grad
    return [self.backward(g, og) for g, og in zip(grad, out)]

  def __call__(self) -> Callable:
    self.first.grad.data = self.backward(self.first.grad.data, self.out.grad.data)
    return self.__call__

class __RSQRT__:
  def __init__(self, first, out) -> None: self.first, self.out = first, out
  def backward(self, grad, out):
    if not isinstance(grad, list):
      # Gradient for rsqrt: -0.5 * x^(-1.5)
      grad += (-0.5 * (out ** -1.5)) * out
      return grad
    return [self.backward(g, og) for g, og in zip(grad, out)]

  def __call__(self) -> Callable:
    self.first.grad.data = self.backward(self.first.grad.data, self.out.grad.data)
    return self.__call__

class __LOG__:
  def __init__(self, first, out) -> None: self.first, self.out = first, out
  def backward(self, grad, out):
    if not isinstance(grad, list):
      grad += (1 / grad) * out
      return grad
    return [self.backward(g, og) for g, og in zip(grad, out)]
  
  def __call__(self) -> Callable:
    self.first.grad.data = self.backward(self.first.grad.data, self.out.grad.data)
    return self.__call__