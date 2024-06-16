from ...helpers.shape import _flatten, _reshape, broadcast_array, re_transpose, get_shape

def _add(a, b):
  if isinstance(a, list) and isinstance(b, list):
    print(len(a), len(b))
    print(a, b)
    assert len(a) == len(b), "Shapes do not match for addition"
    return [_add(ai, bi) for ai, bi in zip(a, b)]
  
  elif isinstance(a, list):
    return [ _add(ai, b) for ai in a]
  
  elif isinstance(b, list):
    return [ _add(a, bi) for bi in b]
  
  else:
    return a + b

class PowBackward:
  def __init__(self, first:list, out:list, exp:float):
    self.first = first
    self.out = out
    self.exp = exp
  
  def backward(self, grad, out, exp):
    if not isinstance(grad, list):
      print(grad, out, exp)
      grad += (exp * out**(exp -1)) * out
      return grad
    return [self.backward(g, og, exp) for g, og in zip(grad, out)]
  
  def __call__(self):
    self.first.grad = self.backward(self.first.grad, self.out.grad, self.exp)
    return self.__call__

# class MatMulBackward:
#   def __init__(self, first, second, out):
#     self.first = first
#     self.second = second
#     self.out = out

#   def backward(self, a, b, out):
#     def matmul_back(a, b, out):
#       def _backward():
#         if a.requires_grad:
#           if a.ndim == 2 and b.ndim == 2:
#             a.grad += _matmul(out.grad, b.transpose(0, -1))
#           else:
#             a.grad += [_backward(a[i], b[i], out[i]) for i in range(len(a.data))]
          
#         if b.requires_grad:
#           if a.ndim == 2 and b.ndim == 2:
#             b.grad += _matmul(a.transpose(0, -1), out.grad)
#           else:
#             b.grad += [_backward(a[i], b[i], out[i]) for i in range(len(b.data))]
#       return _backward
#     return matmul_back(a, b, out)

#   def __call__(self):
#     self.first.grad = self.backward(self.first.grad, self.out.grad, self.second.data)
#     self.second.grad = self.backward(self.second.grad, self.out.grad, self.first.data)
#     return self.__call__

class SumBackward:
  def __init__(self, first, axis, keepdim, out):
    self.first = first
    self.axis = axis
    self.keepdim = keepdim
    self.out = out

  def backward(self, grad):
    if self.axis is None:
      grad = [grad for _ in _flatten(self.first.data)]
    else:
      shape = list(self.first.shape)
      if not self.keepdim:
        shape[self.axis] = 1
      grad = _reshape(grad, shape)
      grad = broadcast_array(grad, self.first.shape)

    return grad

  def __call__(self):
    self.first.grad = self.backward(self.out.grad)
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