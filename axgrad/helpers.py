import math, functools
from ._core import DType, lib
from ctypes import c_int, c_float

class ShapeHelp:
  def get_shape(data: list) -> list: return [len(data)] + ShapeHelp.get_shape(data[0]) if isinstance(data, list) else []
  def flatten(data: list) -> list: return [item for sub in data for item in ShapeHelp.flatten(sub)] if isinstance(data, list) else [data]
  def get_size(shape: tuple) -> int: return math.prod(shape) if shape else 1
  def transpose_shape(shape: list) -> list: return shape if len(shape) == 1 else [shape[1], shape[0]] if len(shape) == 2 else [shape[0], shape[2], shape[1]]
  def reshape_list(flat: list, shape: tuple) -> list:
    if len(shape) == 1: return flat[:shape[0]]
    step = len(flat) // shape[0]
    return [ShapeHelp.reshape_list(flat[i*step:(i+1)*step], shape[1:]) for i in range(shape[0])]

  def get_strides(shape: tuple) -> list:
    if not shape: return []
    strides, acc = [], 1
    for s in reversed(shape[1:]):
      acc *= s
      strides.insert(0, acc)
    strides.append(1)
    return strides

  def process_shape(shape: tuple) -> list:
    s = list(shape[0]) if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else list(shape)
    return s, math.prod(s), len(s), (c_int * len(s))(*s)

  def is_broadcastable(shape_a: tuple, shape_b: tuple) -> bool:
    longer, shorter = (shape_a, shape_b) if len(shape_a) >= len(shape_b) else (shape_b, shape_a)
    return all(s == l or s == 1 or l == 1 for s, l in zip(shorter[::-1], longer[len(longer)-len(shorter):][::-1]))

  def broadcast_shapes(shape_a: tuple, shape_b: tuple) -> tuple:
    longer, shorter = (shape_a, shape_b) if len(shape_a) >= len(shape_b) else (shape_b, shape_a)
    padded = (1,) * (len(longer) - len(shorter)) + shorter
    return tuple(max(a, b) for a, b in zip(longer, padded))


class DtypeHelp:
  dtype_map = { "float32": DType.FLOAT32, "float64": DType.FLOAT64, "int8": DType.INT8, "int16": DType.INT16, "int32": DType.INT32, "int64": DType.INT64, "uint8": DType.UINT8, "uint16": DType.UINT16, "uint32": DType.UINT32, "uint64": DType.UINT64, "bool": DType.BOOL }
  type_dtypes = ["int8","int16","int32","int64","long","float32","float64","double","uint8","uint16","uint32","uint64","bool"]
  def _parse_dtype(dtype: str) -> int:
    if dtype not in DtypeHelp.dtype_map:
      raise ValueError(f"Unsupported dtype: {dtype}. Supported dtypes: {DtypeHelp.type_dtypes}")
    return DtypeHelp.dtype_map[dtype]

def _mk_cidx(indices: list): return (c_int * len(indices))(*indices)

def _flat_to_nd(flat_i: int, shape: tuple) -> list:
  strides = ShapeHelp.get_strides(shape)
  indices = []
  rem = flat_i
  for stride in strides:
    indices.append(rem // stride)
    rem %= stride
  return indices

def _to_scalar(value) -> float:
  from .tensor import Tensor
  if isinstance(value, Tensor): return float(value.tolist())
  if isinstance(value, (list, tuple)): return float(value[0])
  return float(value)

def _set_item_tensor(self, key, value):
  from .tensor import Tensor
  if self.ndim == 0: raise TypeError("0-d tensor cannot be indexed")

  if isinstance(key, int):
    if key < 0: key += self.shape[0]
    if key < 0 or key >= self.shape[0]: raise IndexError(f"Index {key} out of bounds")
    if self.ndim == 1: lib.set_item_tensor(self.data, _mk_cidx([key]), c_float(_to_scalar(value)))
    else:
      sub_shape = self.shape[1:]
      expected = math.prod(sub_shape)

      if isinstance(value, Tensor):
        flat = value.tolist()
        if isinstance(flat, list): flat = ShapeHelp.flatten(flat)
        else: flat = [flat]
      elif isinstance(value, (list, tuple)): flat = ShapeHelp.flatten(list(value))
      else: raise ValueError("Cannot assign scalar to multi-dimensional slice")

      if len(flat) != expected: raise ValueError("Slice size mismatch")
      for i, val in enumerate(flat): lib.set_item_tensor(self.data, _mk_cidx([key] + _flat_to_nd(i, sub_shape)), c_float(float(val)))

  elif isinstance(key, tuple):
    if len(key) > self.ndim: raise IndexError("Too many indices")
    lib.set_item_tensor(self.data, _mk_cidx(list(key) + [0]*(self.ndim-len(key))), c_float(_to_scalar(value)))
  else: raise TypeError("Index must be int or tuple")

def _iter_item_tensor(self):
  if self.ndim == 0: raise TypeError("Iteration over 0-d tensor")
  for i in range(self.shape[0]): yield self[i]

def _get_item_tensor(self, key):
  if self.ndim == 0: raise TypeError("0-d tensor cannot be indexed")
  if isinstance(key, int):
    if key < 0: key += self.shape[0]
    if key < 0 or key >= self.shape[0]: raise IndexError("Index out of bounds")
    if self.ndim == 1: return lib.get_item_tensor(self.data, _mk_cidx([key]))

    def recurse(base, shape):
      if len(shape) == 1: return [lib.get_item_tensor(self.data, _mk_cidx(base+[j])) for j in range(shape[0])]
      return [recurse(base+[i], shape[1:]) for i in range(shape[0])]
    return recurse([key], list(self.shape[1:]))

  elif isinstance(key, tuple):
    if len(key) > self.ndim: raise IndexError("Too many indices")
    return lib.get_item_tensor(self.data, _mk_cidx(list(key) + [0]*(self.ndim-len(key))))
  else: raise TypeError("Index must be int or tuple")