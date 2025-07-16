from .._core import CTensor, lib, DType
from ..helpers import ShapeHelp, DtypeHelp
from ..autograd.functions import *
from typing import *
from ctypes import c_float

def add_tensor_ops(self, other):
  from ..tensor import Tensor
  other_tensor = other if isinstance(other, Tensor) else Tensor([other] if isinstance(other, (int, float)) else other, self.dtype)
  if self.shape != other.shape:
    if ShapeHelp.is_broadcastable(self.shape, other.shape): result_ptr = lib.add_broadcasted_tensor(self.data, other.data).contents
    else: raise ValueError(f"Shapes {self.shape} & {other.shape} are incompatible for broadcasting")
  else: result_ptr = lib.add_scalar_tensor(self.data, c_float(other)).contents if isinstance(other, (int, float)) else lib.add_tensor(self.data, other_tensor.data).contents
  out = Tensor(result_ptr, self.dtype, self.requires_grad or (isinstance(other, Tensor) and other.requires_grad))
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if out.requires_grad: out.grad_fn = AddBackwards(self, other_tensor if isinstance(other, Tensor) else other)
  return out

def sub_tensor_ops(self, other):
  from ..tensor import Tensor
  other_tensor = other if isinstance(other, Tensor) else Tensor([other] if isinstance(other, (int, float)) else other, self.dtype)
  if self.shape != other.shape:
    if ShapeHelp.is_broadcastable(self.shape, other.shape): result_ptr = lib.sub_broadcasted_tensor(self.data, other.data).contents
    else: raise ValueError(f"Shapes {self.shape} & {other.shape} are incompatible for broadcasting")
  else: result_ptr = lib.sub_scalar_tensor(self.data, c_float(other)).contents if isinstance(other, (int, float)) else lib.sub_tensor(self.data, other_tensor.data).contents
  out = Tensor(result_ptr, self.dtype, self.requires_grad or (isinstance(other, Tensor) and other.requires_grad))
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if out.requires_grad: out.grad_fn = SubBackwards(self, other_tensor if isinstance(other, Tensor) else other)
  return out

def mul_tensor_ops(self, other):
  from ..tensor import Tensor
  other_tensor = other if isinstance(other, Tensor) else Tensor([other] if isinstance(other, (int, float)) else other, self.dtype)
  if self.shape != other.shape:
    if ShapeHelp.is_broadcastable(self.shape, other.shape): result_ptr = lib.mul_broadcasted_tensor(self.data, other.data).contents
    else: raise ValueError(f"Shapes {self.shape} & {other.shape} are incompatible for broadcasting")
  else: result_ptr = lib.mul_scalar_tensor(self.data, c_float(other)).contents if isinstance(other, (int, float)) else lib.mul_tensor(self.data, other_tensor.data).contents
  out = Tensor(result_ptr, self.dtype, self.requires_grad or (isinstance(other, Tensor) and other.requires_grad))
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if out.requires_grad: out.grad_fn = SubBackwards(self, other_tensor if isinstance(other, Tensor) else other)
  return out

def div_tensor_ops(self, other):
  from ..tensor import Tensor
  other_tensor = other if isinstance(other, Tensor) else Tensor([other] if isinstance(other, (int, float)) else other, self.dtype)
  if self.shape != other.shape:
    if ShapeHelp.is_broadcastable(self.shape, other.shape): result_ptr = lib.div_broadcasted_tensor(self.data, other.data).contents
    else: raise ValueError(f"Shapes {self.shape} & {other.shape} are incompatible for broadcasting")
  else: result_ptr = lib.div_scalar_tensor(self.data, c_float(other)).contents if isinstance(other, (int, float)) else lib.div_tensor(self.data, other_tensor.data).contents
  out = Tensor(result_ptr, self.dtype, self.requires_grad or (isinstance(other, Tensor) and other.requires_grad))
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if out.requires_grad: out.grad_fn = SubBackwards(self, other_tensor if isinstance(other, Tensor) else other)
  return out

def matmul_tensor_ops(self, other):
  from ..tensor import Tensor
  other = other if isinstance(other, (CTensor, Tensor)) else Tensor(other, self.dtype)
  if self.ndim <= 2 and other.ndim <= 2: result_ptr = lib.matmul_tensor(self.data, other.data).contents
  elif self.ndim == 3 and other.ndim == 3 and self.shape[0] == other.shape[0]: result_ptr = lib.batch_matmul_tensor(self.data, other.data).contents
  else: result_ptr = lib.broadcasted_matmul_tensor(self.data, other.data).contents
  out = Tensor(result_ptr, self.dtype, self.requires_grad or other.requires_grad)
  shape, ndim, size = lib.out_shape(out.data), self.ndim, lib.out_size(out.data)
  out.shape, out.ndim, out.size = tuple([shape[i] for i in range(ndim)]), ndim, size
  out.strides = ShapeHelp.get_strides(out.shape)
  if out.requires_grad: out.grad_fn = MatmulBackwards(self, other)
  return out

def dot_tensor_ops(self, other):
  from ..tensor import Tensor
  other = other if isinstance(other, (CTensor, Tensor)) else Tensor(other, self.dtype)
  if self.ndim <= 2 and other.ndim <= 2:
    result_ptr = lib.dot_tensor(self.data, other.data).contents
  elif self.ndim == 3 and other.ndim == 3 and self.shape[0] == other.shape[0]:
    result_ptr = lib.batch_dot_tensor(self.data, other.data).contents
  out = Tensor(result_ptr, self.dtype, self.requires_grad or other.requires_grad)
  shape, ndim, size = lib.out_shape(result_ptr), out.data.ndim, lib.out_size(result_ptr)
  out.shape, out.ndim, out.size = tuple([shape[i] for i in range(ndim)]), ndim, size
  out.strides = ShapeHelp.get_strides(out.shape)
  if out.requires_grad: out.grad_fn = DotBackwards(self, other)
  return out

def pow_tensor_ops(self, exp):
  from ..tensor import Tensor
  if isinstance(exp, (int, float)): result_ptr = lib.pow_tensor(self.data, c_float(exp)).contents
  out = Tensor(result_ptr, self.dtype, self.requires_grad)
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if self.requires_grad: out.grad_fn = PowBackwards(self, exp)
  return out

def rpow_tensor_ops(self, base):
  from ..tensor import Tensor
  if isinstance(base, (int, float)): result_ptr = lib.pow_scalar(c_float(base), self.data).contents
  else: raise NotImplementedError("__rpow__ with Tensor base not implemented yet")
  out = Tensor(result_ptr, self.dtype, self.requires_grad)
  out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
  if self.requires_grad: out.grad_fn = RPowBackwards(base, self)
  return out

def radd_tensor_ops(self, other):
  from ..tensor import Tensor
  return self + other

def rsub_tensor_ops(self, other):
  from ..tensor import Tensor
  return Tensor([other], self.dtype, self.requires_grad) - self

def rmul_tensor_ops(self, other):
  from ..tensor import Tensor
  return self * other

def rdiv_tensor_ops(self, other):
  from ..tensor import Tensor
  return Tensor([other], self.dtype, self.requires_grad) / self