from ...helpers.ops import matmul
from ...helpers.shape import transpose, get_shape, reshape

def sum_to_shape(grad, shape):
  for i, (grad_dim, shape_dim) in enumerate(zip(get_shape(grad), shape)):
    if grad_dim > shape_dim:
      grad = sum(grad, axis=i)
  return grad

class PowBackward:
  def __init__(self, first:list, out:list, exp:float):
    self.first, self.exp, self.out = first, exp, out
  
  def backward(self, grad, out, exp):
    if not isinstance(grad, list):
      grad += (exp * out**(exp -1)) * out
      return grad
    return [self.backward(g, og, exp) for g, og in zip(grad, out)]
  
  def __call__(self):
    self.first.grad = self.backward(self.first.grad, self.out.grad, self.exp)
    return self.__call__

# class MatmulBackward:
#   def __init__(self, first, second, out):
#     self.first = first
#     self.second = second
#     self.out = out

#   def backward(self, grad, A, B):
#     grad_first = matmul(grad, transpose(B))
#     grad_second = matmul(transpose(A), grad)
#     return grad_first, grad_second

#   def __call__(self):
#     grad_first, grad_second = self.backward(self.out.grad, self.first.data, self.second.data)
#     if self.first.grad is None:
#       self.first.grad = grad_first
#     else:
#       self.first.grad = [a + b for a, b in zip(self.first.grad, grad_first)]

#     if self.second.grad is None:
#       self.second.grad = grad_second
#     else:
#       self.second.grad = [a + b for a, b in zip(self.second.grad, grad_second)]

#     return self.__call__()

class MatmulBackward:
  def __init__(self, first, second, out):
    self.first, self.second, self.out = first, second, out

  def backward(self, grad, out):
    if not isinstance(grad, list):
      grad += out
      return grad
    return [self.backward(g, og) for g, og in zip(grad, out)]

  def __call__(self):
    grad_A = matmul(self.out.grad, transpose(self.second.data))
    grad_B = matmul(transpose(self.first.data), self.out.grad)

    if get_shape(self.first.data) != get_shape(grad_A):
      grad_A = sum_to_shape(grad_A, get_shape(self.first.data))
    if get_shape(self.second.data) != get_shape(grad_B):
      grad_B = sum_to_shape(grad_B, get_shape(self.second.data))

    self.first.grad = self.backward(self.first.grad, grad_A)
    self.second.grad = self.backward(self.second.grad, grad_B)

    return self.__call__

class SumBackward:
  def __init__(self, first, out, axis=None, keepdims=False):
    self.first = first
    self.out = out
    self.axis = axis
    self.keepdims = keepdims

  def expand_gradient(self, grad, original_shape, axis, keepdims):
    # Expanding gradient back to original shape if axis is specified
    if axis is None:
      return [[grad]] * original_shape[0] if keepdims else [grad] * original_shape[0]

    expanded_grad = grad
    for _ in range(len(original_shape) - len(grad) if not keepdims else 0):
      expanded_grad = [expanded_grad] * original_shape[axis]
    return expanded_grad

  def backward(self, grad, out):
    if self.axis is None:
      # If axis is None, sum over all elements, we need to broadcast the gradient to the original shape
      return self.expand_gradient(grad, get_shape(self.first.data), self.axis, self.keepdims)
    else:
      # If axis is specified, we sum along that axis, so we need to expand the gradient along that axis
      expanded_grad = self.expand_gradient(grad, get_shape(self.first.data), self.axis, self.keepdims)
      return expanded_grad

  def __call__(self):
    # Accumulate gradients for the summed input
    self.first.grad = self.backward(self.first.grad, self.out.grad)
    return self.__call__()

class TransposeBackward:
  def __init__(self, first, dim0, dim1, out):
    self.first, self.dim0, self.dim1, self.out = first, dim0, dim1, out

  def backward(self, grad, out):
    return transpose(data=grad)

  def __call__(self):
    self.first.grad = self.backward(self.first.data, self.first.grad, self.out.grad)
    return self.__call__

class ReshapeBackward:
  def __init__(self, one, out, new_shape) -> None:
    self.one, self.out, self.new_shape = one, out, new_shape
  
  def backward(self, grad, out):
    grad = reshape(grad, self.new_shape)
    return grad

  def __call__(self, grad):
    self.first.grad = self.backward(self.first.grad, self.out.grad)