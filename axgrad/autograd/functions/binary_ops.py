from ...helpers.ops import matmul
from ...helpers.shape import transpose

class __ADD__:
  def __init__(self, first, second, out) -> None: self.first, self.second, self.out = first, second, out
  def backward(self, grad, out):
    if not isinstance(grad, list):
      grad += out
      return grad
    return [self.backward(g, og) for g, og, in zip(grad, out)]
  
  def __call__(self):
    self.first.grad = self.backward(self.first.grad, self.out.grad)
    self.second.grad = self.backward(self.second.grad, self.out.grad)
    return self.__call__

class __MUL__:
  def __init__(self, first, second, out) -> None: self.first, self.second, self.out = first, second, out
  def backward(self, grad, out, mul):
    if not isinstance(grad, list):
      grad += out * mul
      return grad
    return [self.backward(g, og, m) for g, og, m in zip(grad, out, mul)]

  def __call__(self):
    self.first.grad = self.backward(self.first.grad, self.out.grad, self.second.data)
    self.second.grad = self.backward(self.second.grad, self.out.grad, self.first.data)
    return self.__call__

class __MATMUL__:
  def __init__(self, first, second, out) -> None: self.first, self.second, self.out = first, second, out
  def backward(self, grad, out, mul):
    if not isinstance(grad, list):
      grad += out * mul
      return grad
    return [self.backward(g, og, m) for g, og, m in zip(grad, out, mul)]

  def __call__(self):
    self.first.grad = self.backward(self.first.grad, self.out.grad, self.second.data)
    self.second.grad = self.backward(self.second.grad, self.out.grad, self.first.data)
    return self.__call__