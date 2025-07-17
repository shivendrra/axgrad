from .._core import CTensor, lib, DType
from ..autograd.unary import *
from typing import *
from ctypes import c_float

def neg_tensor_ops(self):
  from ..tensor import Tensor
  result_pointer = lib.neg_tensor(self.data).contents
  out = Tensor(result_pointer, self.dtype, self.requires_grad)
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if out.requires_grad: out.grad_fn = NegBackwards(self)
  return (setattr(out, "grad", self.grad), setattr(out, "hooks", self.hooks), setattr(out, "grad_fn", self.grad_fn), out)[3] if self.requires_grad else out

def sign_tensor_ops(self):
  from ..tensor import Tensor
  result_pointer = lib.sign_tensor(self.data).contents
  out = Tensor(result_pointer, self.dtype, self.requires_grad)
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  return (setattr(out, "grad", self.grad), setattr(out, "hooks", self.hooks), setattr(out, "grad_fn", self.grad_fn), out)[3] if self.requires_grad else out

def log_tensor_ops(self):
  from ..tensor import Tensor
  result_pointer = lib.log_tensor(self.data).contents
  out = Tensor(result_pointer, self.dtype, self.requires_grad)
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if self.requires_grad: out.grad_fn = LogBackwards(self)
  return out

def exp_tensor_ops(self):
  from ..tensor import Tensor
  result_pointer = lib.exp_tensor(self.data).contents
  out = Tensor(result_pointer, self.dtype, self.requires_grad)
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if self.requires_grad: out.grad_fn = ExpBackwards(self)
  return out

def abs_tensor_ops(self):
  from ..tensor import Tensor
  result_pointer = lib.abs_tensor(self.data).contents
  out = Tensor(result_pointer, self.dtype, self.requires_grad)
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if self.requires_grad: out.grad_fn = AbsBackwards(self)
  return out

def sqrt_tensor_ops(self):
  from ..tensor import Tensor
  result_pointer = lib.sqrt_tensor(self.data).contents
  out = Tensor(result_pointer, self.dtype, self.requires_grad)
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if self.requires_grad: out.grad_fn = SqrtBackwards(self)
  return out