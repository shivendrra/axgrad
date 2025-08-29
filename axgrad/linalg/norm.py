from .._core import lib, DType
from ..autograd.norm import *
from typing import *
from ctypes import c_float
from ..tensor import Tensor

def clip(self, max_val: float):
  out = Tensor(lib.clip_tensor(self.data, c_float(max_val)).contents, self.dtype, self.requires_grad)
  out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
  if self.requires_grad: out.grad_fn = ClipBackwards(self, max_val)
  return out

def clamp(self, min_val: float, max_val: float):
  out = Tensor(lib.clamp_tensor(self.data, c_float(min_val), c_float(max_val)).contents, self.dtype, self.requires_grad)
  out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
  if self.requires_grad: out.grad_fn = ClampBackwards(self, min_val, max_val)
  return out

def mm_norm(self):
  out = Tensor(lib.mm_norm_tensor(self.data).contents, self.dtype, self.requires_grad)
  out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
  if self.requires_grad: out.grad_fn = MMNormBackwards(self)
  return out

def std_norm(self):
  out = Tensor(lib.std_norm_tensor(self.data).contents, self.dtype, self.requires_grad)
  out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
  if self.requires_grad: out.grad_fn = StdNormBackwards(self)
  return out

def rms_norm(self):
  out = Tensor(lib.rms_norm_tensor(self.data).contents, self.dtype, self.requires_grad)
  out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
  if self.requires_grad: out.grad_fn = RMSNormBackwards(self)
  return out

def l1_norm(self):
  out = Tensor(lib.l1_norm_tensor(self.data).contents, self.dtype, self.requires_grad)
  out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
  if self.requires_grad: out.grad_fn = L1NormBackwards(self)
  return out

def l2_norm(self):
  out = Tensor(lib.l2_norm_tensor(self.data).contents, self.dtype, self.requires_grad)
  out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
  if self.requires_grad: out.grad_fn = L2NormBackwards(self)
  return out

def unit_norm(self):
  out = Tensor(lib.unit_norm_tensor(self.data).contents, self.dtype, self.requires_grad)
  out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
  if self.requires_grad: out.grad_fn = UnitNormBackwards(self)
  return out

def robust_norm(self):
  out = Tensor(lib.robust_norm_tensor(self.data).contents, self.dtype, self.requires_grad)
  out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
  if self.requires_grad: out.grad_fn = RobustNormBackwards(self)
  return out