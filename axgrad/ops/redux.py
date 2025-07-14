from .._core import CTensor, lib, DType
from ..helpers import ShapeHelp, DtypeHelp
from ..autograd.functions import *
from typing import *
from ctypes import c_float, c_bool, c_int

def sum_tensor_ops(self, axis: int=-1, keepdims: bool=False):
  from ..tensor import Tensor
  out = Tensor(lib.sum_tensor(self.data, c_int(axis), c_bool(keepdims)).contents, self.dtype, self.requires_grad)
  if axis == -1: out.shape, out.size, out.ndim = (1,) if keepdims else (), 1, 1 if keepdims else 0
  else:
    new_shape = list(self.shape)
    if keepdims: new_shape[axis] = 1
    else: new_shape.pop(axis)
    out.shape = tuple(new_shape)
    out.size, out.ndim, out.strides = 1 if not new_shape else eval('*'.join(map(str, new_shape))), len(new_shape), ShapeHelp.get_strides(out.shape) if out.shape else []
  if self.requires_grad: out.grad_fn = SumBackwards(self, axis, keepdims)
  return out

def mean_tensor_ops(self, axis: int=-1, keepdims: bool=False):
  from ..tensor import Tensor
  out = Tensor(lib.mean_tensor(self.data, c_int(axis), c_bool(keepdims)).contents, self.dtype, self.requires_grad)
  if axis == -1: out.shape, out.size, out.ndim = (1,) if keepdims else (), 1, 1 if keepdims else 0
  else:
    new_shape = list(self.shape)
    if keepdims: new_shape[axis] = 1
    else: new_shape.pop(axis)
    out.shape = tuple(new_shape)
    out.size, out.ndim, out.strides = 1 if not new_shape else eval('*'.join(map(str, new_shape))), len(new_shape), ShapeHelp.get_strides(out.shape) if out.shape else []
  if self.requires_grad: out.grad_fn = MeanBackwards(self, axis, keepdims)
  return out

def min_tensor_ops(self, axis: int=-1, keepdims: bool=False):
  from ..tensor import Tensor
  out = Tensor(lib.min_tensor(self.data, c_int(axis), c_bool(keepdims)).contents, self.dtype, self.requires_grad)
  if axis == -1: out.shape, out.size, out.ndim = (1,) if keepdims else (), 1, 1 if keepdims else 0
  else:
    new_shape = list(self.shape)
    if keepdims: new_shape[axis] = 1
    else: new_shape.pop(axis)
    out.shape = tuple(new_shape)
    out.size, out.ndim, out.strides = 1 if not new_shape else eval('*'.join(map(str, new_shape))), len(new_shape), ShapeHelp.get_strides(out.shape) if out.shape else []
  if self.requires_grad: out.grad_fn = MinBackwards(self, axis, keepdims)
  return out

def max_tensor_ops(self, axis: int=-1, keepdims: bool=False):
  from ..tensor import Tensor
  out = Tensor(lib.max_tensor(self.data, c_int(axis), c_bool(keepdims)).contents, self.dtype, self.requires_grad)
  if axis == -1: out.shape, out.size, out.ndim = (1,) if keepdims else (), 1, 1 if keepdims else 0
  else:
    new_shape = list(self.shape)
    if keepdims: new_shape[axis] = 1
    else: new_shape.pop(axis)
    out.shape = tuple(new_shape)
    out.size, out.ndim, out.strides = 1 if not new_shape else eval('*'.join(map(str, new_shape))), len(new_shape), ShapeHelp.get_strides(out.shape) if out.shape else []
  if self.requires_grad: out.grad_fn = MaxBackwards(self, axis, keepdims)
  return out

def var_tensor_ops(self, axis: int=-1, ddof: int=0):
  from ..tensor import Tensor
  out = Tensor(lib.min_tensor(self.data, c_int(axis), c_int(ddof)).contents, self.dtype, self.requires_grad)
  if axis == -1: out.shape, out.size, out.ndim = (), 1, 0
  else:
    new_shape = list(self.shape)
    new_shape.pop(axis)
    out.shape = tuple(new_shape)
    out.size, out.ndim, out.strides = 1 if not new_shape else eval('*'.join(map(str, new_shape))), len(new_shape), ShapeHelp.get_strides(out.shape) if out.shape else []
  if self.requires_grad: out.grad_fn = VarBackwards(self, axis, None, ddof)
  return out

def std_tensor_ops(self, axis: int=-1, ddof: int=0):
  from ..tensor import Tensor
  out = Tensor(lib.std_tensor(self.data, c_int(axis), c_int(ddof)).contents, self.dtype, self.requires_grad)
  if axis == -1: out.shape, out.size, out.ndim = (), 1, 0
  else:
    new_shape = list(self.shape)
    new_shape.pop(axis)
    out.shape = tuple(new_shape)
    out.size, out.ndim, out.strides = 1 if not new_shape else eval('*'.join(map(str, new_shape))), len(new_shape), ShapeHelp.get_strides(out.shape) if out.shape else []
  if self.requires_grad: out.grad_fn = StdBackwards(self, axis, None, ddof)
  return out

def register_redux_ops():
  from ..tensor import Tensor
  Tensor.sum = sum_tensor_ops
  Tensor.mean = mean_tensor_ops
  Tensor.max = max_tensor_ops
  Tensor.min = min_tensor_ops
  Tensor.var = var_tensor_ops
  Tensor.std = std_tensor_ops