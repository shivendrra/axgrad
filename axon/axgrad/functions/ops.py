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
    self.out = self.out

  def __call__(self):
    return self.__call__