from .._core import CTensor, lib, DType
from ..helpers import ShapeHelp, DtypeHelp
from ..autograd.functions import *
from typing import *
from ctypes import c_float

def transpose_tensor_ops(self):
  from ..tensor import Tensor
  assert self.ndim <= 3, ".transpose() ops limited to 3-d tensor"
  out = Tensor(lib.transpose_tensor(self.data).contents, self.dtype, self.requires_grad)
  out.shape, out.size, out.ndim = tuple(ShapeHelp.transpose_shape(self.shape)), self.size, self.ndim
  out.strides = ShapeHelp.get_strides(out.shape)
  if self.requires_grad: out.grad_fn = TransposeBackwards(self)
  return out

def flatten_tensor_ops(self):
  from ..tensor import Tensor
  out = Tensor(lib.flatten_tensor(self.data).contents, self.dtype, self.requires_grad)
  out.shape, out.size, out.ndim = (self.size, ), self.size, 1
  out.strides = ShapeHelp.get_strides(out.shape)
  if self.requires_grad: out.grad_fn = FlatBackwards(self)
  return out

def reshape_tensor_ops(self):
  from ..tensor import Tensor
  if isinstance(new_shape, tuple): new_shape = list(new_shape)
  new_size, ndim = 1, len(new_shape)
  for dim in new_shape: new_size *= dim
  if new_size != self.size: raise ValueError(f"Cannot reshape Tensor of size {self.size} into shape {new_shape}")
  out = Tensor(lib.reshape_tensor(self.data, (c_int * ndim)(*new_shape), c_int(ndim)).contents, self.dtype, self.requires_grad)
  out.shape, out.size, out.ndim, out.strides = tuple(new_shape), self.size, ndim, ShapeHelp.get_strides(new_shape)
  if self.requires_grad: out.grad_fn = ReshapeBackwards(self)
  return out

def register_shape_ops():
  from ..tensor import Tensor
  Tensor.transpose = transpose_tensor_ops
  Tensor.flatten = flatten_tensor_ops
  Tensor.reshape = reshape_tensor_ops