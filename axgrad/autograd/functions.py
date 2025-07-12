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

class LogBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad / self.input[0]]

class AbsBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad * self.input[0].sign()]

class ExpBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad * self.input[0].exp()]

class SqrtBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad * (0.5 / self.input[0].sqrt())]

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

class SinBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): 
    from ..tensor import Tensor
    out = Tensor(lib.sin_backwards(self.input[0].data).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
    return [grad * out]

class SinhBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.sinh_backwards(self.input[0].data).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
    return [grad * out]

class CosBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.cos_backwards(self.input[0].data).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
    return [grad * out]

class CoshBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.cosh_backwards(self.input[0].data).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
    return [grad * out]

class TanBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.tan_backwards(self.input[0].data).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
    return [grad * out]

class TanhBackwards:
  def __init__(self, x, out): self.input, self.output = [x], out
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.tanh_backwards(self.output.data).contents, self.output.dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.output.shape, self.output.ndim, self.output.size, self.output.strides
    return [grad * out]

class SigmoidBackwards:
  def __init__(self, x, out): self.input, self.output = [x], out
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.sigmoid_backwards(self.output.data).contents, self.output.dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.output.shape, self.output.ndim, self.output.size, self.output.strides
    return [grad * out]

class ReluBackwards:
  def __init__(self, x, out): self.input, self.output = [x], out
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.relu_backwards(self.output.data).contents, self.output.dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.output.shape, self.output.ndim, self.output.size, self.output.strides
    return [grad * out]

class EluBackwards:
  def __init__(self, x, out, alpha=1e-5): self.input, self.output, self.alpha = [x], out, alpha
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.elu_backwards(self.input[0].data, c_float(self.alpha)).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
    return [grad * out]

class LeakyReluBackwards:
  def __init__(self, x, out, eps=1e-5): self.input, self.output, self.eps = [x], out, eps
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.leaky_relu_backwards(self.input[0].data, c_float(self.eps)).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
    return [grad * out]

class GeluBackwards:
  def __init__(self, x, out): self.input, self.output = [x], out
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.gelu_backwards(self.input[0].data).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
    return [grad * out]

class SwishBackwards:
  def __init__(self, x, out, beta=1e-5): self.input, self.output, self.beta = [x], out, beta
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.swish_backwards(self.input[0].data, c_float(self.beta)).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
    return [grad * out]

class SiluBackwards:
  def __init__(self, x, out): self.input, self.output = [x], out
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.silu_backwards(self.input[0].data).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
    return [grad * out]

class SoftplusBackwards:
  def __init__(self, x, out): self.input, self.output = [x], out
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.softplus_backwards(self.input[0].data).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
    return [grad * out]

class TransposeBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad.transpose()]

class FlatBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad.reshape(self.input[0].shape)]

class ReshapeBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad.reshape(self.input[0].shape)]