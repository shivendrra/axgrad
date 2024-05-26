from ...helpers.shape import transpose as tp

class PowBackward:
  def __init__(self, first:list, out:list, exp:float):
    self.first = first
    self.out = out
    self.exp = exp
  
  def backward(self, grad, out, exp):
    if not isinstance(grad, list):
      grad += (exp * out**(exp -1)) * out
      return grad
    return [self.backward(g, og, exp) for g, og in zip(grad, out)]
  
  def __call__(self):
    self.one.grad = self.backward(self.one.grad, self.out.grad, self.exp)
    return self.__call__

class MatMulBackward:
  def __init__(self, first:list, second:list, out:list):
    self.first = first
    self.second = second
    self.out = out

  def backward(self, grad, out, other, transpose=False):
    if not isinstance(grad, list):
      if not transpose:
        grad += grad * other
      else:
        grad += grad * tp(other)
      return grad
    return [self.backward(g, og, o, transpose) for g, og, o in zip(grad, out, other)]

  def __call__(self):
    self.first.grad = self.backward(self.first.grad, self.out.grad, self.second.data, transpose=True)
    self.second.grad = self.backward(self.second.grad, self.out.grad, self.first.data)
    return self.__call__