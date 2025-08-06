from typing import *
from ctypes import c_int, c_float, c_double
from .._core import CTensor, lib, DType
from ..tensor import Tensor
from ..helpers import DtypeHelp, ShapeHelp

def det(a: Tensor, dtype: DType = 'float32', requires_grad:bool = False) -> Tensor:
  a = a if isinstance(a, Tensor, requires_grad) else Tensor(a, 'float32', requires_grad)
  if a.ndim == 2:
    ptr = lib.det_tensor(a.data).contents
    out_shape, out_size, out_ndim, out_strides = (), 1, 0, ()
  elif a.ndim == 3:
    ptr = lib.batched_det_tensor(a.data).contents
    out_shape, out_size, out_ndim, out_strides = a.shape[:-2], a.size // (a.shape[-1] * a.shape[-2]), a.ndim - 2, ShapeHelp.get_strides(a.shape[:-2])
  else: raise ValueError("Can't compute determinant for 3d > ndims")
  out = Tensor(ptr, dtype if dtype is not None else a.dtype, requires_grad)
  return (setattr(out, "shape", out_shape), setattr(out, "ndim", out_ndim), setattr(out, "size", out_size), setattr(out, "strides", out_strides), out)[4]

def lu(a: Tensor, dtype: DType = 'float32', requires_grad:bool = False) -> Tensor:
  a = a if isinstance(a, Tensor, requires_grad) else Tensor(a, 'float32', requires_grad)
  result_ptr = lib.lu_tensor(a.data) if a.ndim == 2 else lib.batched_lu_Tensor(a.data)
  if a.ndim == 2:
    l_shape, u_shape = (a.shape[0], a.shape[0]), (a.shape[0], a.shape[1])
    l_size, u_size = a.shape[0] * a.shape[0], a.shape[0] * a.shape[1]
  else:
    l_shape, u_shape = a.shape[:-2] + (a.shape[-2], a.shape[-2]), a.shape[:-2] + (a.shape[-2], a.shape[-1])
    l_size, u_size = (a.size // a.shape[-1]) * a.shape[-2], a.size // a.shape[-1]
  l_out, u_out = Tensor(result_ptr[0].contents, dtype or a.dtype, requires_grad or a.requires_grad), Tensor(result_ptr[1].contents, dtype or a.dtype, requires_grad or a.requires_grad)
  for out, shape, size in [(l_out, l_shape, l_size), (u_out, u_shape, u_size)]:
    out.shape, out.ndim, out.size, out.strides = shape, len(shape), size, ShapeHelp.get_strides(shape)
  return [l_out, u_out]

def qr(a: Tensor, dtype: DType = 'float32', requires_grad:bool = False) -> Tensor:
  a = a if isinstance(a, Tensor, requires_grad) else Tensor(a, 'float32', requires_grad)
  result_ptr = lib.qr_tensor(a.data) if a.ndim == 2 else lib.batched_qr_Tensor(a.data)
  if a.ndim == 2:
    q_shape, r_shape = (a.shape[0], a.shape[0]), (a.shape[0], a.shape[1])
    q_size, r_size = a.shape[0] * a.shape[0], a.shape[0] * a.shape[1]
  else:
    q_shape, r_shape = a.shape[:-2] + (a.shape[-2], a.shape[-2]), a.shape[:-2] + (a.shape[-2], a.shape[-1])
    q_size, r_size = (a.size // a.shape[-1]) * a.shape[-2], a.size // a.shape[-1]
  q_out, r_out = Tensor(result_ptr[0].contents, dtype or a.dtype, requires_grad or a.requires_grad), Tensor(result_ptr[1].contents, dtype or a.dtype, requires_grad or a.requires_grad)
  for out, shape, size in [(q_out, q_shape, q_size), (r_out, r_shape, r_size)]:
    out.shape, out.ndim, out.size, out.strides = shape, len(shape), size, ShapeHelp.get_strides(shape)
  return [q_out, r_out]

def svd(a: Tensor, dtype: DType = 'float32', requires_grad: bool=False) -> Tensor:
  a = a if isinstance(a, Tensor) else Tensor(a, 'float32')
  result_tuple = lib.svd_tensor(a.data)

  u_ptr = result_tuple[0].contents
  s_ptr = result_tuple[1].contents
  vt_ptr = result_tuple[2].contents
  u, s, vt = Tensor(u_ptr, dtype or a.dtype, requires_grad or a.requires_grad), Tensor(s_ptr, dtype or a.dtype, requires_grad or a.requires_grad), Tensor(vt_ptr, dtype or a.dtype, requires_grad or requires_grad)
  m, n = a.shape[-2], a.shape[-1]
  min_mn = min(m, n)

  if a.ndim == 2:
    u_shape, u_size, u_ndim, u_strides = (m, m), m * m, 2, ShapeHelp.get_strides((m, m))
    s_shape, s_size, s_ndim, s_strides = (min_mn, ), min_mn, 1, (1, )
    vt_shape, vt_size, vt_ndim, vt_strides = (n, n), n * n, 2, ShapeHelp.get_strides((n, n))
  else:
    batch_shape, batch_size = a.shape[:-2], 1
    for dim in batch_shape: batch_size *= dim
    u_shape, s_shape, vt_shape = batch_shape + (m, m), batch_shape + (min_mn,), batch_shape + (n, n)
    u_size, s_size, vt_size = batch_size * m * m, batch_size * min_mn, batch_size * n * n
    u_ndim, s_ndim, vt_ndim = a.ndim, a.ndim - 1, a.ndim
    u_strides, s_strides, vt_strides = ShapeHelp.get_strides(u_shape), ShapeHelp.get_strides(s_shape), ShapeHelp.get_strides(vt_shape)

  u.shape, u.size, u.ndim, u.strides = u_shape, u_size, u_ndim, u_strides
  s.shape, s.size, s.ndim, s.strides = s_shape, s_size, s_ndim, s_strides
  vt.shape, vt.size, vt.ndim, vt.strides = vt_shape, vt_size, vt_ndim, vt_strides
  return u, s, vt

def cholesky(a: Tensor, dtype: DType = 'float32', requires_grad: bool=False) -> Tensor:
  a = a if isinstance(a, Tensor, requires_grad) else Tensor(a, 'float32', requires_grad)
  ptr = lib.cholesky_tensor(a.data).contents
  out = Tensor(ptr, dtype or a.dtype, requires_grad or a.requires_grad)
  out.shape, out.size, out.ndim, out.strides = a.shape, a.size, a.ndim, a.strides
  return out

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