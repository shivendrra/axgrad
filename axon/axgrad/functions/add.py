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
    return self.__call__