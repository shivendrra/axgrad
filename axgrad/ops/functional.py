from .._core import CTensor, lib, DType
from ..autograd.functions import *
from typing import *
from ctypes import c_float

def sin_tensor_ops(self):
  from ..tensor import Tensor
  result_ptr = lib.sin_tensor(self.data).contents
  out = Tensor(result_ptr, self.dtype, self.requires_grad)
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if self.requires_grad: out.grad_fn = SinBackwards(self)
  return out

def cos_tensor_ops(self):
  from ..tensor import Tensor
  result_ptr = lib.cos_tensor(self.data).contents
  out = Tensor(result_ptr, self.dtype, self.requires_grad)
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if self.requires_grad: out.grad_fn = CosBackwards(self)
  return out

def tan_tensor_ops(self):
  from ..tensor import Tensor
  result_ptr = lib.tan_tensor(self.data).contents
  out = Tensor(result_ptr, self.dtype, self.requires_grad)
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if self.requires_grad: out.grad_fn = TanBackwards(self)
  return out

def sinh_tensor_ops(self):
  from ..tensor import Tensor
  result_ptr = lib.sinh_tensor(self.data).contents
  out = Tensor(result_ptr, self.dtype, self.requires_grad)
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if self.requires_grad: out.grad_fn = SinhBackwards(self)
  return out

def cosh_tensor_ops(self):
  from ..tensor import Tensor
  result_ptr = lib.cosh_tensor(self.data).contents
  out = Tensor(result_ptr, self.dtype, self.requires_grad)
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if self.requires_grad: out.grad_fn = CoshBackwards(self)
  return out

def tanh_tensor_ops(self):
  from ..tensor import Tensor
  result_ptr = lib.tanh_tensor(self.data).contents
  out = Tensor(result_ptr, self.dtype, self.requires_grad)
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if self.requires_grad: out.grad_fn = TanhBackwards(self, out)
  return out

def sigmoid_tensor_ops(self):
  from ..tensor import Tensor
  result_ptr = lib.sigmoid_tensor(self.data).contents
  out = Tensor(result_ptr, self.dtype, self.requires_grad)
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if self.requires_grad: out.grad_fn = SigmoidBackwards(self, out)
  return out

def relu_tensor_ops(self):
  from ..tensor import Tensor
  result_ptr = lib.relu_tensor(self.data).contents
  out = Tensor(result_ptr, self.dtype, self.requires_grad)
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if self.requires_grad: out.grad_fn = ReluBackwards(self, out)
  return out

def elu_tensor_ops(self, alpha: float= 1e-5):
  from ..tensor import Tensor
  out = Tensor(lib.elu_tensor(self.data, c_float(alpha)).contents, self.dtype, self.requires_grad)
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if self.requires_grad: out.grad_fn = EluBackwards(self, out, alpha)  # Passing alpha parameter
  return out

def leaky_relu_tensor_ops(self, eps: float= 1e-5):
  from ..tensor import Tensor
  out = Tensor(lib.leaky_relu_tensor(self.data, c_float(eps)).contents, self.dtype, self.requires_grad)
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if self.requires_grad: out.grad_fn = LeakyReluBackwards(self, out, eps)  # Passing eps parameter
  return out

def gelu_tensor_ops(self):
  from ..tensor import Tensor
  result_ptr = lib.gelu_tensor(self.data).contents
  out = Tensor(result_ptr, self.dtype, self.requires_grad)
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if self.requires_grad: out.grad_fn = GeluBackwards(self, out)
  return out

def swish_tensor_ops(self, beta: float= 0.5):
  from ..tensor import Tensor
  out = Tensor(lib.swish_tensor(self.data, c_float(beta)).contents, self.dtype, self.requires_grad)
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if self.requires_grad: out.grad_fn = SwishBackwards(self, out, beta)  # Passing beta parameter
  return out

def softplus_tensor_ops(self):
  from ..tensor import Tensor
  result_ptr = lib.softplus_tensor(self.data).contents
  out = Tensor(result_ptr, self.dtype, self.requires_grad)
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if self.requires_grad: out.grad_fn = SoftplusBackwards(self, out)
  return out

def silu_tensor_ops(self):
  from ..tensor import Tensor
  result_ptr = lib.silu_tensor(self.data).contents
  out = Tensor(result_ptr, self.dtype, self.requires_grad)
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if self.requires_grad: out.grad_fn = SiluBackwards(self, out)
  return out