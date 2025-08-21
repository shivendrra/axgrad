from ._core import CTensor, lib, DType
from ctypes import c_float, c_size_t, c_int
from typing import *
from .tensor import Tensor
from .helpers import DtypeHelp, ShapeHelp

def zeros_like(arr: Tensor) -> Tensor:
  ptr = lib.zeros_like_tensor(arr.data if isinstance(arr, Tensor) else arr).contents; out = Tensor(ptr)
  return (setattr(out, "shape", arr.shape), setattr(out, "ndim", arr.ndim), setattr(out, "size", arr.size), setattr(out, "strides", arr.strides), out)[4]

def ones_like(arr: Tensor) -> Tensor:
  ptr = lib.ones_like_tensor(arr.data if isinstance(arr, Tensor) else arr).contents; out = Tensor(ptr)
  return (setattr(out, "shape", arr.shape), setattr(out, "ndim", arr.ndim), setattr(out, "size", arr.size), setattr(out, "strides", arr.strides), out)[4]

def zeros(*shape: Union[list, tuple], dtype: DType="float32") -> Tensor:
  s, sz, nd, sa = ShapeHelp.process_shape(shape)
  out = Tensor(lib.zeros_tensor(sa, c_size_t(sz), c_size_t(nd), c_int(DtypeHelp._parse_dtype(dtype))).contents, dtype, False)
  return (setattr(out, "shape", tuple(s)), setattr(out, "ndim", nd), setattr(out, "size", sz), setattr(out, "strides", ShapeHelp.get_strides(s)), out)[4]

def ones(*shape: Union[list, tuple], dtype: DType="float32") -> Tensor:
  s, sz, nd, sa = ShapeHelp.process_shape(shape)
  out = Tensor(lib.ones_tensor(sa, c_size_t(sz), c_size_t(nd), c_int(DtypeHelp._parse_dtype(dtype))).contents, dtype, False)
  return (setattr(out, "shape", tuple(s)), setattr(out, "ndim", nd), setattr(out, "size", sz), setattr(out, "strides", ShapeHelp.get_strides(s)), out)[4]

def randn(*shape: Union[list, tuple], dtype: DType="float32") -> Tensor:
  s, sz, nd, sa = ShapeHelp.process_shape(shape)
  out = Tensor(lib.randn_tensor(sa, c_size_t(sz), c_size_t(nd), c_int(DtypeHelp._parse_dtype(dtype))).contents, dtype, False)
  return (setattr(out, "shape", tuple(s)), setattr(out, "ndim", nd), setattr(out, "size", sz), setattr(out, "strides", ShapeHelp.get_strides(s)), out)[4]

def randint(low: int, high: int, *shape: Union[list, tuple], dtype: DType="int32") -> Tensor:
  s, sz, nd, sa = ShapeHelp.process_shape(shape)
  out = Tensor(lib.randint_tensor(c_int(int(low)), c_int(int(high)), sa, c_size_t(sz), c_size_t(nd), c_int(DtypeHelp._parse_dtype(dtype))).contents, dtype, False)
  return (setattr(out, "shape", tuple(s)), setattr(out, "ndim", nd), setattr(out, "size", sz), setattr(out, "strides", ShapeHelp.get_strides(s)), out)[4]

def uniform(low: int, high: int, *shape: Union[list, tuple], dtype: DType="float32") -> Tensor:
  s, sz, nd, sa = ShapeHelp.process_shape(shape)
  out = Tensor(lib.uniform_tensor(c_int(int(low)), c_int(int(high)), sa, c_size_t(sz), c_size_t(nd), c_int(DtypeHelp._parse_dtype(dtype))).contents, dtype, False)
  return (setattr(out, "shape", tuple(s)), setattr(out, "ndim", nd), setattr(out, "size", sz), setattr(out, "strides", ShapeHelp.get_strides(s)), out)[4]

def fill(fill_val: Union[float, int], *shape: Union[list, tuple], dtype: DType="float32") -> Tensor:
  s, sz, nd, sa = ShapeHelp.process_shape(shape)
  out = Tensor(lib.fill_tensor(c_float(fill_val), sa, c_size_t(sz), c_size_t(nd), c_int(DtypeHelp._parse_dtype(dtype))).contents, dtype, False)
  return (setattr(out, "shape", tuple(s)), setattr(out, "ndim", nd), setattr(out, "size", sz), setattr(out, "strides", ShapeHelp.get_strides(s)), out)[4]

def linspace(start: float, step: float, end: float, *shape: Union[list, tuple], dtype: DType= "float32") -> Tensor:
  s, sz, nd, sa = ShapeHelp.process_shape(shape)
  out = Tensor(lib.linspace_tensor(c_float(start), c_float(step), c_float(end), sa, c_size_t(sz), c_size_t(nd), c_int(DtypeHelp._parse_dtype(dtype))).contents, dtype, False)
  return (setattr(out, "shape", tuple(s)), setattr(out, "ndim", nd), setattr(out, "size", sz), setattr(out, "strides", ShapeHelp.get_strides(s)), out)[4]

def arange(start: float, stop: float, step: float=1.0, dtype: DType= "float32") -> Tensor:
  if step == 0.0: raise ValueError("Step cannot be zero")
  if (step > 0 and start >= stop) or (step < 0 and start <= stop): raise ValueError("Invalid arange parameters: no values in range")
  c_array = lib.arange_tensor(c_float(start), c_float(stop), c_float(step), c_int(DtypeHelp._parse_dtype(dtype)))
  sz = lib.out_size(c_array)
  out = Tensor(c_array.contents, dtype, False)
  return (setattr(out, "shape", (sz,)), setattr(out, "ndim", 1), setattr(out, "size", sz), setattr(out, "strides", ShapeHelp.get_strides((sz,))), out)[4]