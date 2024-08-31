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

  def backward(self, grad, out_grad, other, transpose=False):
    if not isinstance(grad, list):
      if not transpose:
        grad += out_grad * other
      else:
        grad += out_grad * list(map(list, zip(*other)))
      return grad
    return [self.backward(g, og, o, transpose) for g, og, o in zip(grad, out_grad, other)]

  def __call__(self):
    self.first.grad = self.backward(self.first.grad, self.out.grad, self.second.data, transpose=True)
    self.second.grad = self.backward(self.second.grad, self.out.grad, self.first.data)
    return self.__call__

class SumBackward:
  def __init__(self, first, out):
    self.first = first
    self.out = out

  def backward(self, grad, out):
    raise NotImplementedError("no Backward function for SumBackward")
  
  def __call__(self):
    self.first.grad = self.backward(self.first.grad, self.out.grad)
    return self.__call__

class TransposeBackward:
  def __init__(self, first, dim0, dim1, out):
    self.first = first
    self.dim0 = dim0
    self.dim1 = dim1
    self.out = out

  def backward(self, grad, out):
    return self._transpose(grad, self.dim0, self.dim1, out)
  
  def _transpose(self, data, dim0, dim1):
    if isinstance(data, list):
      if len(data) == 0:
        return data
      if isinstance(data[0], list):
        transposed = [list(row) for row in zip(*data)]
        return [self._transpose(row, dim0, dim1) for row in transposed]
      else:
        return [list(row) for row in zip(*data)]
    return data

  def __call__(self):
    self.first.grad = self.backward(self.first.data, self.first.grad, self.out.grad)
    return self.__call__