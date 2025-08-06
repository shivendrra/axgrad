from ..tensor import CTensor, lib, DType
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

def reshape_tensor_ops(self, new_shape):
  from ..tensor import Tensor
  if isinstance(new_shape, tuple): new_shape = list(new_shape)
  new_size, ndim = 1, len(new_shape)
  for dim in new_shape: new_size *= dim
  if new_size != self.size: raise ValueError(f"Cannot reshape Tensor of size {self.size} into shape {new_shape}")
  out = Tensor(lib.reshape_tensor(self.data, (c_int * ndim)(*new_shape), c_int(ndim)).contents, self.dtype, self.requires_grad)
  out.shape, out.size, out.ndim, out.strides = tuple(new_shape), self.size, ndim, ShapeHelp.get_strides(new_shape)
  if self.requires_grad: out.grad_fn = ReshapeBackwards(self)
  return out

def unsqueeze_tensor_ops(self, axis):
  from ..tensor import Tensor
  result_ptr = lib.expand_dims_tensor(self.data, c_int(axis)).contents
  out = Tensor(result_ptr, self.dtype, self.requires_grad)
  new_shape = list(self.shape)
  new_shape.insert(axis, 1)
  out.shape = tuple(new_shape)
  out.size, out.ndim, out.strides = self.size, len(out.shape), ShapeHelp.get_strides(out.shape)
  return out

def squeeze_tensor_ops(self, axis):
  from ..tensor import Tensor
  result_ptr = lib.squeeze_tensor(self.data, c_int(axis)).contents
  out = Tensor(result_ptr, self.dtype, self.requires_grad)
  if axis == -1: new_shape = [dim for dim in self.shape if dim != 1]      # Remove all dimensions of size 1
  else: # Remove specific axis if it has size 1
    if self.shape[axis] != 1: raise ValueError(f"Cannot squeeze axis {axis} with size {self.shape[axis]}")
    new_shape = list(self.shape); new_shape.pop(axis)
  out.shape = tuple(new_shape) if new_shape else (1,)
  out.size, out.ndim, out.strides = self.size, len(out.shape), ShapeHelp.get_strides(out.shape)
  return out

def contiguous_tensor_ops(self):
  from ..tensor import Tensor
  out = Tensor(lib.contiguous_tensor(self.data).contents, self.dtype)
  out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
  return out

def make_contiguous_tensor_ops(self) -> None:
  lib.make_contiguous_inplace_tensor(self.data)
  self.strides = ShapeHelp.get_strides(self.shape)  # updating strides since they may have changed

def view_tensor_ops(self):
  from ..tensor import Tensor
  out = Tensor(lib.view_tensor(self.data).contents, self.dtype)
  out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
  return out