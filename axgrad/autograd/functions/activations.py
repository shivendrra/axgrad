from ...ops.functionals import *

class __RELU__:
  def __init__(self, first, out) -> None: self.first, self.out = first, out
  def backward(self, grad, data):
    if not isinstance(grad, list):
      grad += relu_derivative(data) * grad
      return grad
    return [self.backward(g, d) for g, d in zip(grad, data)]
  def __call__(self):
    self.first.grad.data = self.backward(self.out.grad.data, self.first.data)
    return self.__call__

class __TANH__:
  def __init__(self, first, out) -> None: self.first, self.out = first, out
  def backward(self, grad, data):
    if not isinstance(grad, list):
      grad += tanh_derivative(data) * grad
      return grad
    return [self.backward(g, d) for g, d in zip(grad, data)]
  def __call__(self):
    self.first.grad.data = self.backward(self.out.grad.data, self.first.data)
    return self.__call__

class __SIGMOID__:
  def __init__(self, first, out) -> None: self.first, self.out = first, out
  def backward(self, grad, data):
    if not isinstance(grad, list):
      grad += sigmoid_derivative(data) * grad
      return grad
    return [self.backward(g, d) for g, d in zip(grad, data)]
  def __call__(self):
    self.first.grad.data = self.backward(self.out.grad.data, self.first.data)
    return self.__call__

class __GELU__:
  def __init__(self, first, out) -> None: self.first, self.out = first, out
  def backward(self, grad, data):
    if not isinstance(grad, list):
      grad += gelu_derivative(data) * grad
      return grad
    return [self.backward(g, d) for g, d in zip(grad, data)]
  def __call__(self):
    self.first.grad.data = self.backward(self.out.grad.data, self.first.data)
    return self.__call__

class __SILU__:
  def __init__(self, first, out) -> None: self.first, self.out = first, out
  def backward(self, grad, data):
    if not isinstance(grad, list):
      grad += silu_derivative(data) * grad
      return grad
    return [self.backward(g, d) for g, d in zip(grad, data)]
  def __call__(self):
    self.first.grad.data = self.backward(self.out.grad.data, self.first.data)
    return self.__call__

class __LRELU__:
  def __init__(self, first, out) -> None: self.first, self.out = first, out
  def backward(self, grad, data):
    if not isinstance(grad, list):
      grad += LeakyRelu_derivative(data) * grad
      return grad
    return [self.backward(g, d) for g, d in zip(grad, data)]
  def __call__(self):
    self.first.grad.data = self.backward(self.out.grad.data, self.first.data)
    return self.__call__

class __SIN__:
  def __init__(self, first, out) -> None: self.first, self.out = first, out
  def backward(self, grad, data):
    if not isinstance(grad, list):
      grad += math.cos(data) * grad
      return grad
    return [self.backward(g, d) for g, d in zip(grad, data)]
  def __call__(self):
    self.first.grad.data = self.backward(self.out.grad.data, self.first.data)
    return self.__call__

class __COS__:
  def __init__(self, first, out) -> None: self.first, self.out = first, out
  def backward(self, grad, data):
    if not isinstance(grad, list):
      grad += -math.sin(data) * grad
      return grad
    return [self.backward(g, d) for g, d in zip(grad, data)]
  def __call__(self):
    self.first.grad.data = self.backward(self.out.grad.data, self.first.data)
    return self.__call__