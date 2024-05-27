from typing import Any
from ...helpers.acitvations import *

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

class tanhBackward:
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

class sigmoidBackward:
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

class geluBackward:
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

class leakyreluBackward:
  def __init__(self, first, out):
    self.first = first
    self.out = out
  
  def backward(self, grad, out, data):
    if not isinstance(grad, list):
      grad += LeakyRELU_derivative(data) * grad
      return grad
    return [self.backward(g, og, d) for g, og, d in zip(grad, out, data)]
  
  def __call__(self):
    self.first.grad = self.backward(self.first.grad, self.out.grad, self.first.data)
    return self.__call__