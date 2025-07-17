from .._core import lib, DType
from ..autograd.norm import *
from typing import *
from ctypes import c_float

def clip_tensor_ops(self, max_val: float):
  from ..tensor import Tensor
  out = Tensor(lib.clip_tensor(self.data, c_float(max_val)).contents, self.dtype, self.requires_grad)
  out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
  if self.requires_grad: out.grad_fn = ClipBackwards(self, max_val)
  return out

def clamp_tensor_ops(self, min_val: float, max_val: float):
  from ..tensor import Tensor
  out = Tensor(lib.clamp_tensor(self.data, c_float(min_val), c_float(max_val)).contents, self.dtype, self.requires_grad)
  out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
  if self.requires_grad: out.grad_fn = ClampBackwards(self, min_val, max_val)
  return out

def mm_norm_tensor_ops(self):
  from ..tensor import Tensor
  out = Tensor(lib.mm_norm_tensor(self.data).contents, self.dtype, self.requires_grad)
  out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
  if self.requires_grad: out.grad_fn = MMNormBackwards(self)
  return out

def std_norm_tensor_ops(self):
  from ..tensor import Tensor
  out = Tensor(lib.std_norm_tensor(self.data).contents, self.dtype, self.requires_grad)
  out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
  if self.requires_grad: out.grad_fn = StdNormBackwards(self)
  return out

def rms_norm_tensor_ops(self):
  from ..tensor import Tensor
  out = Tensor(lib.rms_norm_tensor(self.data).contents, self.dtype, self.requires_grad)
  out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
  if self.requires_grad: out.grad_fn = RMSNormBackwards(self)
  return out

def l1_norm_tensor_ops(self):
  from ..tensor import Tensor
  out = Tensor(lib.l1_norm_tensor(self.data).contents, self.dtype, self.requires_grad)
  out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
  if self.requires_grad: out.grad_fn = L1NormBackwards(self)
  return out

def l2_norm_tensor_ops(self):
  from ..tensor import Tensor
  out = Tensor(lib.l2_norm_tensor(self.data).contents, self.dtype, self.requires_grad)
  out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
  if self.requires_grad: out.grad_fn = L2NormBackwards(self)
  return out

def unit_norm_tensor_ops(self):
  from ..tensor import Tensor
  out = Tensor(lib.unit_norm_tensor(self.data).contents, self.dtype, self.requires_grad)
  out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
  if self.requires_grad: out.grad_fn = UnitNormBackwards(self)
  return out

def robust_norm_tensor_ops(self):
  from ..tensor import Tensor
  out = Tensor(lib.robust_norm_tensor(self.data).contents, self.dtype, self.requires_grad)
  out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
  if self.requires_grad: out.grad_fn = RobustNormBackwards(self)
  return out

def register_norm_ops():
  from ..tensor import Tensor
  Tensor.mm_norm = mm_norm_tensor_ops
  Tensor.std_norm = std_norm_tensor_ops
  Tensor.rms_norm = rms_norm_tensor_ops
  Tensor.l1_norm = l1_norm_tensor_ops
  Tensor.l2_norm = l2_norm_tensor_ops
  Tensor.unit_norm = unit_norm_tensor_ops
  Tensor.robust_norm = robust_norm_tensor_ops