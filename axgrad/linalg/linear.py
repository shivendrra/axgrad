from typing import *
from ctypes import c_int, c_float, c_double
from .._core import CTensor, lib, DType
from ..tensor import Tensor
from ..helpers import DtypeHelp, ShapeHelp

def linear(a: Tensor, b: Tensor, bias: Tensor = None, dtype: DType = 'float32', requires_grad: bool=False) -> Tensor:
  a, b = a if isinstance(a, Tensor) else Tensor(a, 'float32', requires_grad), b if isinstance(b, Tensor) else Tensor(b, 'float32', requires_grad)
  if bias is not None:
    ptr = lib.linear_transform_tensor(a.data, b.data, bias.data if isinstance(bias, Tensor) else bias).contents
  else:
    if a.ndim == 1: ptr = lib.linear_1d_tensor(a.data, b.data, None).contents
    elif a.ndim == 2: ptr = lib.linear_2d_tensor(a.data, b.data, None).contents
    else: ptr = lib.linear_transform_tensor(a.data, b.data, None).contents

  out = Tensor(ptr, dtype if dtype is not None else a.dtype or b.dtype, requires_grad)
  if a.ndim == 1 and b.ndim == 1:
    out_shape, out_size, out_ndim, out_strides = (), 1, 0, ()
  elif a.ndim == 1 and b.ndim == 2:
    out_shape, out_size, out_ndim, out_strides = (b.shape[1],), b.shape[1], 1, (1,)
  elif a.ndim == 2 and b.ndim == 1:
    out_shape, out_size, out_ndim, out_strides = (a.shape[0],), a.shape[0], 1, (1,)
  elif a.ndim == 2 and b.ndim == 2:
    out_shape, out_size, out_ndim, out_strides = (a.shape[0], b.shape[1]), a.shape[0] * b.shape[1], 2, ShapeHelp.get_strides((a.shape[0], b.shape[1]))
  else:
    out_shape, out_size, out_ndim, out_strides = a.shape, a.size, a.ndim, a.strides  
  return (setattr(out, "shape", out_shape), setattr(out, "ndim", out_ndim), setattr(out, "size", out_size), setattr(out, "strides", out_strides), out)[4]