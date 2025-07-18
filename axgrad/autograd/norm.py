from .._core import lib
from ctypes import c_float

class ClipBackwards:
  def __init__(self, x, max_val): self.input, self.max_val = [x], max_val
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.clip_backwards(self.input[0].data, grad.data, c_float(self.max_val)).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
    return [out]

class ClampBackwards:
  def __init__(self, x, min_val, max_val):  self.input, self.min_val, self.max_val = [x], min_val, max_val
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.clamp_backwards(self.input[0].data, grad.data, c_float(self.min_val), c_float(self.max_val)).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
    return [out]

class MMNormBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.mm_norm_backwards(self.input[0].data, grad.data).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
    return [out]

class RMSNormBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.rms_norm_backwards(self.input[0].data, grad.data).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
    return [out]

class L1NormBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.l1_norm_backwards(self.input[0].data, grad.data).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
    return [out]

class L2NormBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.l2_norm_backwards(self.input[0].data, grad.data).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
    return [out]

class StdNormBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.std_norm_backwards(self.input[0].data, grad.data).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
    return [out]

class UnitNormBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.unit_norm_backwards(self.input[0].data, grad.data).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
    return [out]

class RobustNormBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad):
    from ..tensor import Tensor
    out = Tensor(lib.robust_norm_backwards(self.input[0].data, grad.data).contents, self.input[0].dtype, False)
    out.shape, out.ndim, out.size, out.strides = self.input[0].shape, self.input[0].ndim, self.input[0].size, self.input[0].strides
    return [out]