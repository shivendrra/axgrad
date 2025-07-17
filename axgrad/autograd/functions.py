import math
from .._core import lib
from ctypes import c_float, c_int

# Keep existing functions that don't have C implementations
class AddBackwards:
  def __init__(self, x, y): self.input = [x, y]
  def backward(self, grad): return [grad, grad]

class SubBackwards:
  def __init__(self, x, y): self.input = [x, y]
  def backward(self, grad): return [grad, -grad]

class MulBackwards:
  def __init__(self, x, y): self.input = [x, y]
  def backward(self, grad): return [self.input[1] * grad, self.input[0] * grad]

class DivBackwards:
  def __init__(self, x, y): self.input = [x, y]
  def backward(self, grad): return [grad * (self.input[1] ** -1), grad * (-self.input[0] / (self.input[1] ** 2))]

class NegBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad.__neg__()]

class PowBackwards:
  def __init__(self, x, exp): self.input = [x, exp]
  def backward(self, grad):
    base, exp = self.input[0], self.input[1]
    if isinstance(base, (int, float)): g_base, g_exp = grad * (base ** (exp - 1)), (grad * base ** exp) * math.log(base)
    else: g_base, g_exp = grad * exp * (base ** (exp - 1)), (grad * base ** exp) * (base.log())
    return [g_base, g_exp]

class RPowBackwards:
  def __init__(self, base, exp): self.input = [base, exp]
  def backward(self, grad): return [None, grad * (self.input[0] ** self.input[1]) * math.log(self.input[0])]


class MatmulBackwards:
  def __init__(self, x, y): self.input = [x, y]
  def backward(self, grad): return [grad @ self.input[1].transpose(), self.input[0].transpose() @ grad]

class DotBackwards:
  def __init__(self, x, y): self.input = [x, y]
  def backward(self, grad): return [grad @ self.input[1].transpose(), self.input[0].transpose() @ grad]

class SumBackwards:
  def __init__(self, x, axis, keepdims): 
    self.input = [x]
    self.axis, self.keepdims = axis, keepdims
    self.o_shape, self.o_ndim, self.o_size = x.shape, x.ndim, x.size

  def backward(self, grad):
    from ..tensor import Tensor

    if grad is None or grad.data is None: raise ValueError("Gradient cannot be None")
    if self.o_size <= 0: raise ValueError("Invalid original tensor size")

    if self.o_ndim == 0: return [grad]      # Handle scalar case (ndim=0) - just return the gradient as is
    shape_array = (c_int * self.o_ndim)(*self.o_shape)
    result_tensor = lib.sum_backwards(grad.data, shape_array, int(self.o_ndim), int(self.o_size), int(self.axis if self.axis is not None else -1)).contents      
    out = Tensor(result_tensor, self.input[0].dtype, False)
    out.shape, out.ndim, out.size = self.o_shape, self.o_ndim, self.o_size
    return [out]

class MeanBackwards:
  def __init__(self, x, axis, keepdims): 
    self.input = [x]
    self.axis, self.keepdims = axis, keepdims
    self.o_shape, self.o_ndim, self.o_size = x.shape, x.ndim, x.size

  def backward(self, grad):
    from ..tensor import Tensor

    if grad is None or grad.data is None: raise ValueError("Gradient cannot be None")
    if self.o_size <= 0: raise ValueError("Invalid original tensor size")
    if self.o_ndim == 0: return [grad]  # Handle scalar case (ndim=0) - just return the gradient as is
    shape_array = (c_int * self.o_ndim)(*self.o_shape)
    result_tensor = lib.mean_backwards(grad.data, shape_array, int(self.o_ndim), int(self.o_size), int(self.axis if self.axis is not None else -1)).contents
    out = Tensor(result_tensor, self.input[0].dtype, False)
    out.shape, out.ndim, out.size = self.o_shape, self.o_ndim, self.o_size
    return [out]

class VarBackwards:
  def __init__(self, x, axis, keepdims, ddof=0): 
    self.input = [x]
    self.axis, self.keepdims, self.ddof = axis, keepdims, ddof
    self.o_shape, self.o_ndim, self.o_size = x.shape, x.ndim, x.size

  def backward(self, grad):
    from ..tensor import Tensor
    shape_array = (c_int * self.o_ndim)(*self.o_shape)
    result_tensor = lib.var_backwards(self.input[0].data, grad.data, shape_array, int(self.o_ndim), int(self.o_size), int(self.axis if self.axis is not None else -1), self.ddof).contents
    out = Tensor(result_tensor, self.input[0].dtype, False)
    out.shape, out.ndim, out.size = self.o_shape, self.o_ndim, self.o_size
    return [out]

class StdBackwards:
  def __init__(self, x, axis, keepdims, ddof=0): 
    self.input = [x]
    self.axis, self.keepdims, self.ddof = axis, keepdims, ddof
    self.o_shape, self.o_ndim, self.o_size = x.shape, x.ndim, x.size

  def backward(self, grad):
    from ..tensor import Tensor
    shape_array = (c_int * self.o_ndim)(*self.o_shape)
    result_tensor = lib.std_backwards(self.input[0].data, grad.data, shape_array, int(self.o_ndim), int(self.o_size), int(self.axis if self.axis is not None else -1), self.ddof).contents
    out = Tensor(result_tensor, self.input[0].dtype, False)
    out.shape, out.ndim, out.size = self.o_shape, self.o_ndim, self.o_size
    return [out]

class TransposeBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad.transpose()]

class FlatBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad.reshape(self.input[0].shape)]

class ReshapeBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad.reshape(self.input[0].shape)]

class ClipBackwards:
  def __init__(self, x, max_val): self.input, self.max_val = [x], max_val
  def backward(self, grad): return [grad * (self.input[0] <= self.max_val)]

class ClampBackwards:
  def __init__(self, x, min_val, max_val):  self.input, self.min_val, self.max_val = [x], min_val, max_val
  def backward(self, grad): return [grad * ((self.input[0] >= self.min_val) * (self.input[0] <= self.max_val))]

class MinBackwards:
  def __init__(self, x, axis, keepdims): 
    self.input, self.axis, self.keepdims = [x], axis, keepdims
    self.o_shape, self.o_ndim, self.o_size = x.shape, x.ndim, x.size
  
  def backward(self, grad):
    if self.o_ndim == 0: return [grad]
    min_vals = self.input[0].min(axis=self.axis, keepdims=True)
    mask = (self.input[0] == min_vals).float()
    if not self.keepdims and self.axis != -1:
      expanded_shape = list(self.o_shape)
      expanded_shape[self.axis] = 1
      grad = grad.reshape(expanded_shape)
    elif not self.keepdims and self.axis == -1: grad = grad.reshape((1,) * self.o_ndim)
    return [grad * mask]

class MaxBackwards:
  def __init__(self, x, axis, keepdims): 
    self.input, self.axis, self.keepdims = [x], axis, keepdims
    self.o_shape, self.o_ndim, self.o_size = x.shape, x.ndim, x.size

  def backward(self, grad):
    if self.o_ndim == 0: return [grad]
    max_vals = self.input[0].max(axis=self.axis, keepdims=True)
    mask = (self.input[0] == max_vals).float()

    if not self.keepdims and self.axis != -1:
      expanded_shape = list(self.o_shape)
      expanded_shape[self.axis] = 1
      grad = grad.reshape(expanded_shape)
    elif not self.keepdims and self.axis == -1: grad = grad.reshape((1,) * self.o_ndim)
    return [grad * mask]