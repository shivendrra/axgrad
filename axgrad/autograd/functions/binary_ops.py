from ...helpers.ops import matmul
from ...helpers.shape import transpose, get_shape

def sum_to_shape(grad, shape):
  for i, (grad_dim, shape_dim) in enumerate(zip(get_shape(grad), shape)):
    if grad_dim > shape_dim:
      grad = sum(grad, axis=i)
  return grad

class __ADD__:
  def __init__(self, first, second, out) -> None: self.first, self.second, self.out = first, second, out
  def backward(self, grad, out):
    if not isinstance(grad, list):
      grad += out
      return grad
    return [self.backward(g, og) for g, og, in zip(grad, out)]
  
  def __call__(self):
    self.first.grad.data = self.backward(self.first.grad.data, self.out.grad.data)
    self.second.grad.data = self.backward(self.second.grad.data, self.out.grad.data)
    return self.__call__

class __MUL__:
  def __init__(self, first, second, out) -> None: self.first, self.second, self.out = first, second, out
  def backward(self, grad, out, mul):
    if not isinstance(grad, list):
      grad += out * mul
      return grad
    return [self.backward(g, og, m) for g, og, m in zip(grad, out, mul)]

  def __call__(self):
    self.first.grad.data = self.backward(self.first.grad.data, self.out.grad.data, self.second.data)
    self.second.grad.data = self.backward(self.second.grad.data, self.out.grad.data, self.first.data)
    return self.__call__

class __MATMUL__:
  def __init__(self, first, second, out) -> None: self.first, self.second, self.out = first, second, out
  def backward(self, grad, out):
    if not isinstance(grad, list):
      grad += out
      return grad
    return [self.backward(g, og) for g, og in zip(grad, out)]

  def __call__(self):
    grad_A = matmul(self.out.grad.data, transpose(self.second.data))
    grad_B = matmul(transpose(self.first.data), self.out.grad.data)

    if get_shape(self.first.data) != get_shape(grad_A):
      grad_A = sum_to_shape(grad_A, get_shape(self.first.data))
    if get_shape(self.second.data) != get_shape(grad_B):
      grad_B = sum_to_shape(grad_B, get_shape(self.second.data))

    self.first.grad.data = self.backward(self.first.grad.data, grad_A)
    self.second.grad.data = self.backward(self.second.grad.data, grad_B)
    return self.__call__

class __POW__:
  def __init__(self, first, out, power) -> None: self.first, self.out, self.power = first, out, power
  def backward(self, grad, out, power):
    if not isinstance(grad, list):
      grad += (power * out ** (power - 1)) * out
      return grad
    return [self.backward(g, og, power) for g, og in zip(grad, out)]
  
  def __call__(self):
    self.first.grad.data = self.backward(self.first.grad.data, self.out.grad.data, self.power)
    return self.__call__