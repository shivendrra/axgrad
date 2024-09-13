from typing import Any
from ...helpers.functionals import *

class ReluBackward:
  def __init__(self, first, out):
    self.first = first
    self.out = out

  def backward(self, grad, out, data):
    if not isinstance(grad, list):
      grad += relu_derivative(data) * grad
      return grad
    return [self.backward(g, og, d) for g, og, d in zip(grad, out, data)]

  def __call__(self):
    self.first.grad = self.backward(self.first.grad, self.out.grad, self.first.data)
    return self.__call__

class TanhBackward:
  def __init__(self, first, out):
    self.first = first
    self.out = out

  def backward(self, grad, out, data):
    if not isinstance(grad, list):
      grad += tanh_derivative(data) * grad
      return grad
    return [self.backward(g, og, d) for g, og, d in zip(grad, out, data)]

  def __call__(self):
    self.first.grad = self.backward(self.first.grad, self.out.grad, self.first.data)
    return self.__call__

class SigmoidBackward:
  def __init__(self, first, out):
    self.first = first
    self.out = out

  def backward(self, grad, out, data):
    if not isinstance(grad, list):
      grad += sigmoid_derivative(data) * grad
      return grad
    return [self.backward(g, og, d) for g, og, d in zip(grad, out, data)]

  def __call__(self):
    self.first.grad = self.backward(self.first.grad, self.out.grad, self.first.data)
    return self.__call__

class GELUBackward:
  def __init__(self, first, out):
    self.first = first
    self.out = out
  
  def backward(self, grad, out, data):
    if not isinstance(grad, list):
      grad += gelu_derivative(data) * grad
      return grad
    return [self.backward(g, og, d) for g, og, d in zip(grad, out, data)]
  
  def __call__(self):
    self.first.grad = self.backward(self.first.grad, self.out.grad, self.first.data)
    return self.__call__

class LeakyRELUBackward:
  def __init__(self, first, out):
    self.first = first
    self.out = out
  
  def backward(self, grad, out, data):
    if not isinstance(grad, list):
      grad += LeakyRelu_derivative(data) * grad
      return grad
    return [self.backward(g, og, d) for g, og, d in zip(grad, out, data)]
  
  def __call__(self):
    self.first.grad = self.backward(self.first.grad, self.out.grad, self.first.data)
    return self.__call__
  
class SiluBackward:
  def __init__(self, first, out):
    self.first = first
    self.out = out
  
  def backward(self, grad, out, data):
    if not isinstance(grad, list):
      grad += silu_derivative(data) * grad
      return grad
    return [self.backward(g, og, d) for g, og, d in zip(grad, out, data)]
  
  def __call__(self):
    self.first.grad = self.backward(self.first.grad, self.out.grad, self.first.data)
    return self.__call__