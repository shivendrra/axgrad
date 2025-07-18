from .._core import lib
from ctypes import c_float

class NegBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad.__neg__()]

class LogBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.log_backwards(self.input[0].data, grad.data).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
    return [out]

class AbsBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.abs_backwards(self.input[0].data, grad.data).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
    return [out]

class ExpBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.exp_backwards(self.input[0].data, grad.data).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
    return [out]

class SqrtBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.sqrt_backwards(self.input[0].data, grad.data).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
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
