from typing import *
from ctypes import c_int, c_float, c_double
from .._core import CTensor, lib, DType
from ..tensor import Tensor
from ..helpers import DtypeHelp, ShapeHelp

def dot(a: Tensor, b: Tensor, dtype: DType = 'float32', requires_grad:bool = False) -> Tensor:
  a, b = a if isinstance(a, Tensor) else Tensor(a, 'float32', requires_grad), b if isinstance(b, Tensor) else Tensor(b, 'float32', requires_grad)
  ptr = lib.vector_dot(a.data, b.data).contents
  out = Tensor(ptr, dtype if dtype is not None else a.dtype, requires_grad)
  return (setattr(out, "shape", ()), setattr(out, "ndim", 0), setattr(out, "size", 1), setattr(out, "strides", ()), out)[4]

def dot_mv(mat: Tensor, vec: Tensor, dtype: DType = 'float32', requires_grad:bool = False) -> Tensor:
  mat, vec = mat if isinstance(mat, Tensor) else Tensor(mat, 'float32', requires_grad), vec if isinstance(vec, Tensor) else Tensor(vec, 'float32', requires_grad)
  ptr = lib.vector_matrix_dot(vec.data, mat.data).contents
  out = Tensor(ptr, dtype if dtype is not None else mat.dtype, requires_grad)
  out_shape, out_size, out_ndim, out_strides = (mat.shape[0],), mat.shape[0], 1, (1,)
  return (setattr(out, "shape", out_shape), setattr(out, "ndim", out_ndim), setattr(out, "size", out_size), setattr(out, "strides", out_strides), out)[4]

def inner(a: Tensor, b: Tensor, dtype: DType = 'float32', requires_grad:bool = False) -> Tensor:
  a, b = a if isinstance(a, Tensor) else Tensor(a, 'float32', requires_grad), b if isinstance(b, Tensor) else Tensor(b, 'float32', requires_grad)
  ptr = lib.vector_inner(a.data, b.data).contents
  out = Tensor(ptr, dtype if dtype is not None else a.dtype, requires_grad)
  return (setattr(out, "shape", ()), setattr(out, "ndim", 0), setattr(out, "size", 1), setattr(out, "strides", ()), out)[4]

def outer(a: Tensor, b: Tensor, dtype: DType = 'float32', requires_grad:bool = False) -> Tensor:
  a, b = a if isinstance(a, Tensor) else Tensor(a, 'float32', requires_grad), b if isinstance(b, Tensor) else Tensor(b, 'float32', requires_grad)
  ptr = lib.vector_outer(a.data, b.data).contents
  out = Tensor(ptr, dtype if dtype is not None else a.dtype, requires_grad)
  out_shape, out_size, out_ndim, out_strides = (a.shape[0], b.shape[0]), a.shape[0] * b.shape[0], 2, ShapeHelp.get_strides((a.shape[0], b.shape[0]))
  return (setattr(out, "shape", out_shape), setattr(out, "ndim", out_ndim), setattr(out, "size", out_size), setattr(out, "strides", out_strides), out)[4]

def cross(a: Tensor, b: Tensor, axis: int=None, dtype: DType = 'float32', requires_grad:bool = False) -> Tensor:
  a, b = a if isinstance(a, Tensor) else Tensor(a, 'float32', requires_grad), b if isinstance(b, Tensor) else Tensor(b, 'float32', requires_grad)
  if a.ndim == 1 and b.ndim == 1:
    ptr = lib.vector_cross(a.data, b.data).contents
  elif a.ndim == 2 and b.ndim == 2 or a.ndim == 3 and b.ndim == 3:
    if axis == None: raise ValueError("Axis value can't be NULL, need an axis value")
    if axis > a.ndim or axis > b.ndim: raise IndexError(f"Can't exceed the ndim. Axis >= ndim in this case: {axis} >= {a.ndim}")
    ptr = lib.vector_cross_axis(a.data, b.data, c_int(axis)).contents
  else:
    raise ValueError(".cross() only supported for 1D, 2D, and 3D vectors")
  out = Tensor(ptr, dtype if dtype is not None else a.dtype, requires_grad)
  out_shape, out_size, out_ndim, out_strides = a.shape, a.size, a.ndim, a.strides
  return (setattr(out, "shape", out_shape), setattr(out, "ndim", out_ndim), setattr(out, "size", out_size), setattr(out, "strides", out_strides), out)[4]