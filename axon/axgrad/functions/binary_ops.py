class AddBackward:
  def __init__(self, first, second, out):
    self.first = first
    self.second = second
    self.out = out

  def backward(self, grad, out):
    if not isinstance(grad, list):
      grad += out
      return grad
    return [self.backward(g, og) for g, og in zip(grad, out)]

  def __call__(self):
    self.first.grad = self.backward(self.first.grad, self.out.grad)
    self.second.grad = self.backward(self.second.grad, self.out.grad)
    print("first grad: ", self.first.grad)
    print("second grad: ", self.second.grad)
    print("self: ", self.first)
    print("second: ", self.second)
    return self.__call__

class MulBackward:
  def __init__(self, first, second, out):
    self.first = first
    self.second = second
    self.out = out

  def backward(self, grad, out, mul):
    if not isinstance(grad, list):
      grad += out * mul
      return grad
    return [self.backward(g, og, m) for g, og, m in zip(grad, out, mul)]

  def __call__(self):
    self.first.grad = self.backward(self.first.grad, self.out.grad, self.second.data)
    self.second.grad = self.backward(self.second.grad, self.out.grad, self.first.data)
    return self.__call__