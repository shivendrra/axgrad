from ._core import CTensor, lib, DType
from ctypes import c_float, c_size_t, c_int
from typing import *
from ._tensor import Tensor
from ._helpers import get_strides

def _process_shape(shape: Union[list, tuple]) -> Tuple[List[int], int, int, Any]:
  shape = list(shape[0]) if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else list(shape)
  size, ndim = 1, len(shape)
  for dim in shape:
    size *= dim
  shape_arr = (c_int * ndim)(*shape)
  return shape, size, ndim, shape_arr

def _parse_dtype(dtype: Union[int, str]) -> int:
  if isinstance(dtype, str):
    return getattr(DType, dtype.upper())
  return dtype

def zeros_like(arr: Tensor) -> Tensor:
  if isinstance(arr, Tensor):
    result_ptr = lib.zeros_like_Tensor(arr.data).contents
  elif isinstance(arr, CTensor):
    result_ptr = lib.zeros_like_Tensor(arr)
  out = Tensor(result_ptr)
  out.shape, out.ndim, out.size, out.strides = arr.shape, arr.ndim, arr.size, arr.strides
  return out

def ones_like(arr: Tensor) -> Tensor:
  if isinstance(arr, Tensor):
    result_ptr = lib.ones_like_Tensor(arr.data).contents
  elif isinstance(arr, CTensor):
    result_ptr = lib.ones_like_Tensor(arr)
  out = Tensor(result_ptr)
  out.shape, out.ndim, out.size, out.strides = arr.shape, arr.ndim, arr.size, arr.strides
  return out

def zeros(*shape: Union[list, tuple, int, float], dtype: DType = DType.FLOAT32) -> Tensor:
  shape, size, ndim, shape_arr = _process_shape(shape)
  parsed_dtype = _parse_dtype(dtype)
  result_ptr = lib.zeros_Tensor(shape_arr, c_size_t(size), c_size_t(ndim), c_int(parsed_dtype)).contents
  out = Tensor(result_ptr)
  out.shape, out.ndim, out.size, out.strides = tuple(shape), ndim, size, get_strides(shape)
  return out

def ones(*shape: Union[list, tuple, int, float], dtype: DType = DType.FLOAT32) -> Tensor:
  shape, size, ndim, shape_arr = _process_shape(shape)
  parsed_dtype = _parse_dtype(dtype)
  result_ptr = lib.ones_Tensor(shape_arr, c_size_t(size), c_size_t(ndim), c_int(parsed_dtype)).contents
  out = Tensor(result_ptr)
  out.shape, out.ndim, out.size, out.strides = tuple(shape), ndim, size, get_strides(shape)
  return out

def randn(*shape: Union[list, tuple], dtype: DType = DType.FLOAT32) -> Tensor:
  shape, size, ndim, shape_arr = _process_shape(shape)
  parsed_dtype = _parse_dtype(dtype)
  result_ptr = lib.randn_Tensor(shape_arr, c_size_t(size), c_size_t(ndim), c_int(parsed_dtype)).contents
  out = Tensor(result_ptr)
  out.shape, out.ndim, out.size, out.strides = tuple(shape), ndim, size, get_strides(shape)
  return out

def randint(low: int, high: int, *shape: Union[list, tuple], dtype: DType = DType.INT32) -> Tensor:
  shape, size, ndim, shape_arr = _process_shape(shape)
  parsed_dtype = _parse_dtype(dtype)
  result_ptr = lib.randint_Tensor(c_int(low), c_int(high), shape_arr, c_size_t(size), c_size_t(ndim), c_int(parsed_dtype)).contents
  out = Tensor(result_ptr)
  out.shape, out.ndim, out.size, out.strides = tuple(shape), ndim, size, get_strides(shape)
  return out

def uniform(low: int, high: int, *shape: Union[list, tuple], dtype: DType = DType.FLOAT32) -> Tensor:
  shape, size, ndim, shape_arr = _process_shape(shape)
  parsed_dtype = _parse_dtype(dtype)
  result_ptr = lib.uniform_Tensor(c_int(low), c_int(high), shape_arr, c_size_t(size), c_size_t(ndim), c_int(parsed_dtype)).contents
  out = Tensor(result_ptr)
  out.shape, out.ndim, out.size, out.strides = tuple(shape), ndim, size, get_strides(shape)
  return out

def fill(fill_val: float, *shape: Union[list, tuple], dtype: DType = DType.FLOAT32) -> Tensor:
  shape, size, ndim, shape_arr = _process_shape(shape)
  parsed_dtype = _parse_dtype(dtype)
  result_ptr = lib.fill_Tensor(c_float(fill_val), shape_arr, c_size_t(size), c_size_t(ndim), c_int(parsed_dtype)).contents
  out = Tensor(result_ptr)
  out.shape, out.ndim, out.size, out.strides = tuple(shape), ndim, size, get_strides(shape)
  return out