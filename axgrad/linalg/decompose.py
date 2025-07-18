from typing import *
from ctypes import c_int, c_float, c_double
from .._core import CTensor, lib, DType
from ..tensor import Tensor
from ..helpers import DtypeHelp, ShapeHelp

def det(a: Tensor, dtype: DType = 'float32', requires_grad:bool = False) -> Tensor:
  a = a if isinstance(a, Tensor, requires_grad) else Tensor(a, 'float32', requires_grad)
  if a.ndim == 2:
    ptr = lib.det_tensor(a.data).contents
    out_shape, out_size, out_ndim, out_strides = (), 0, 1, ()
  elif a.ndim == 3:
    ptr = lib.batched_det_tensor(a.data).contents
    out_shape, out_size, out_ndim, out_strides = a.shape[:-2], a.size // (a.shape[-1] * a.shape[-2]), a.ndim - 2, ShapeHelp.get_strides(a.shape[:-2])
  else: raise ValueError("Can't compute determinant for 3d > ndims")
  out = Tensor(ptr, dtype if dtype is not None else a.dtype, requires_grad)
  return (setattr(out, "shape", out_shape), setattr(out, "ndim", out_ndim), setattr(out, "size", out_size), setattr(out, "strides", out_strides), out)[4]

def eign(a: Tensor, dtype: DType = 'float32', requires_grad:bool = False) -> Tensor:
  a = a if isinstance(a, Tensor, requires_grad) else Tensor(a, 'float32', requires_grad)
  if a.ndim == 2:
    ptr = lib.eig_tensor(a.data).contents
    out_shape, out_size, out_ndim, out_strides = (a.shape[0],), a.shape[0], 1, (1,)
  elif a.ndim == 3:
    ptr = lib.batched_eig_tensor(a.data).contents
    out_shape, out_size, out_ndim, out_strides = a.shape[:-1], a.size // a.shape[-1], a.ndim - 1, ShapeHelp.get_strides(a.shape[:-1])
  else: raise ValueError("Can't compute eigenvalues for 3d > ndims")
  out = Tensor(ptr, dtype if dtype is not None else a.dtype, requires_grad)
  return (setattr(out, "shape", out_shape), setattr(out, "ndim", out_ndim), setattr(out, "size", out_size), setattr(out, "strides", out_strides), out)[4]

def eignv(a: Tensor, dtype: DType = 'float32', requires_grad:bool = False) -> Tensor:
  a = a if isinstance(a, Tensor, requires_grad) else Tensor(a, 'float32', requires_grad)
  if a.ndim == 2:
    ptr = lib.eigv_tensor(a.data).contents
    out_shape, out_size, out_ndim, out_strides = a.shape, a.size, a.ndim, a.strides
  elif a.ndim == 3:
    ptr = lib.batched_eigv_tensor(a.data).contents
    out_shape, out_size, out_ndim, out_strides = a.shape, a.size, a.ndim, a.strides
  else: raise ValueError("Can't compute eigenvectors for 3d > ndims")
  out = Tensor(ptr, dtype if dtype is not None else a.dtype, requires_grad)
  return (setattr(out, "shape", out_shape), setattr(out, "ndim", out_ndim), setattr(out, "size", out_size), setattr(out, "strides", out_strides), out)[4]

def eignh(a: Tensor, dtype: DType = 'float32', requires_grad:bool = False) -> Tensor:
  a = a if isinstance(a, Tensor, requires_grad) else Tensor(a, 'float32', requires_grad)
  if a.ndim == 2:
    ptr = lib.eigh_tensor(a.data).contents
    out_shape, out_size, out_ndim, out_strides = (a.shape[0],), a.shape[0], 1, (1,)
  elif a.ndim == 3:
    ptr = lib.batched_eigh_tensor(a.data).contents
    out_shape, out_size, out_ndim, out_strides = a.shape[:-1], a.size // a.shape[-1], a.ndim - 1, ShapeHelp.get_strides(a.shape[:-1])
  else: raise ValueError("Can't compute hermitian eigenvalues for 3d > ndims")
  out = Tensor(ptr, dtype if dtype is not None else a.dtype, requires_grad)
  return (setattr(out, "shape", out_shape), setattr(out, "ndim", out_ndim), setattr(out, "size", out_size), setattr(out, "strides", out_strides), out)[4]

def eignhv(a: Tensor, dtype: DType = 'float32', requires_grad:bool = False) -> Tensor:
  a = a if isinstance(a, Tensor, requires_grad) else Tensor(a, 'float32', requires_grad)
  if a.ndim == 2:
    ptr = lib.eighv_tensor(a.data).contents
    out_shape, out_size, out_ndim, out_strides = a.shape, a.size, a.ndim, a.strides
  elif a.ndim == 3:
    ptr = lib.batched_eighv_tensor(a.data).contents
    out_shape, out_size, out_ndim, out_strides = a.shape, a.size, a.ndim, a.strides
  else: raise ValueError("Can't compute hermitian eigenvectors for 3d > ndims")
  out = Tensor(ptr, dtype if dtype is not None else a.dtype, requires_grad)
  return (setattr(out, "shape", out_shape), setattr(out, "ndim", out_ndim), setattr(out, "size", out_size), setattr(out, "strides", out_strides), out)[4]