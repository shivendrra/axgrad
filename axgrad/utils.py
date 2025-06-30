from ._core import CTensor, lib, DType
from ctypes import c_float, c_size_t, c_int
from typing import *
from .tensor import Tensor
from .helpers import DtypeHelp, ShapeHelp

def zeros_like(arr):
  ptr = lib.zeros_like_tensor(arr.data if isinstance(arr, Tensor) else arr).contents; out = Tensor(ptr)
  return (setattr(out, "shape", arr.shape), setattr(out, "ndim", arr.ndim), setattr(out, "size", arr.size), setattr(out, "strides", arr.strides), out)[4]

def ones_like(arr):
  ptr = lib.ones_like_tensor(arr.data if isinstance(arr, Tensor) else arr).contents; out = Tensor(ptr)
  return (setattr(out, "shape", arr.shape), setattr(out, "ndim", arr.ndim), setattr(out, "size", arr.size), setattr(out, "strides", arr.strides), out)[4]

def zeros(*shape, dtype=DType.FLOAT32):
  s, sz, nd, sa = ShapeHelp.process_shape(shape)
  out = Tensor(lib.zeros_tensor(sa, c_size_t(sz), c_size_t(nd), c_int(DtypeHelp._parse_dtype(dtype))).contents)
  return (setattr(out, "shape", tuple(s)), setattr(out, "ndim", nd), setattr(out, "size", sz), setattr(out, "strides", ShapeHelp.get_strides(s)), out)[4]

def ones(*shape, dtype=DType.FLOAT32):
  s, sz, nd, sa = ShapeHelp.process_shape(shape)
  out = Tensor(lib.ones_tensor(sa, c_size_t(sz), c_size_t(nd), c_int(DtypeHelp._parse_dtype(dtype))).contents)
  return (setattr(out, "shape", tuple(s)), setattr(out, "ndim", nd), setattr(out, "size", sz), setattr(out, "strides", ShapeHelp.get_strides(s)), out)[4]

def randn(*shape, dtype=DType.FLOAT32):
  s, sz, nd, sa = ShapeHelp.process_shape(shape)
  out = Tensor(lib.randn_tensor(sa, c_size_t(sz), c_size_t(nd), c_int(DtypeHelp._parse_dtype(dtype))).contents)
  return (setattr(out, "shape", tuple(s)), setattr(out, "ndim", nd), setattr(out, "size", sz), setattr(out, "strides", ShapeHelp.get_strides(s)), out)[4]

def randint(low, high, *shape, dtype=DType.INT32):
  s, sz, nd, sa = ShapeHelp.process_shape(shape)
  out = Tensor(lib.randint_tensor(c_int(low), c_int(high), sa, c_size_t(sz), c_size_t(nd), c_int(DtypeHelp._parse_dtype(dtype))).contents)
  return (setattr(out, "shape", tuple(s)), setattr(out, "ndim", nd), setattr(out, "size", sz), setattr(out, "strides", ShapeHelp.get_strides(s)), out)[4]

def uniform(low, high, *shape, dtype=DType.FLOAT32):
  s, sz, nd, sa = ShapeHelp.process_shape(shape)
  out = Tensor(lib.uniform_tensor(c_int(low), c_int(high), sa, c_size_t(sz), c_size_t(nd), c_int(DtypeHelp._parse_dtype(dtype))).contents)
  return (setattr(out, "shape", tuple(s)), setattr(out, "ndim", nd), setattr(out, "size", sz), setattr(out, "strides", ShapeHelp.get_strides(s)), out)[4]

def fill(fill_val, *shape, dtype=DType.FLOAT32):
  s, sz, nd, sa = ShapeHelp.process_shape(shape)
  out = Tensor(lib.fill_tensor(c_float(fill_val), sa, c_size_t(sz), c_size_t(nd), c_int(DtypeHelp._parse_dtype(dtype))).contents)
  return (setattr(out, "shape", tuple(s)), setattr(out, "ndim", nd), setattr(out, "size", sz), setattr(out, "strides", ShapeHelp.get_strides(s)), out)[4]