import math, functools
from ._core import DType, lib
from ctypes import c_int, c_float

class ShapeHelp:
  # helper class for shape related functions
  def get_shape(data: list) -> list: return [len(data), ] + ShapeHelp.get_shape(data[0]) if isinstance(data, list) else []
  def flatten(data: list) -> list: return [item for sublist in data for item in ShapeHelp.flatten(sublist)] if isinstance(data, list) else [data]
  def get_size(shape: tuple) -> int: return 1 if not shape else shape[0] * ShapeHelp.get_size(shape[1:])
  def transpose_shape(shape: list) -> list: return shape if len(shape) == 1 else [shape[1], shape[0]] if len(shape) == 2 else [shape[0], shape[2], shape[1]]
  def reshape_list(flat_list:list, shape: tuple) -> list: return flat_list[:shape[0]] if len(shape) == 1 else [ShapeHelp.reshape_list(flat_list[i * (len(flat_list) // shape[0]):(i + 1) * (len(flat_list) // shape[0])], shape[1:]) for i in range(shape[0])]
  def get_strides(shape: tuple) -> list: return [] if not shape else [1] if len(shape) == 1 else [functools.reduce(lambda a, b: a * b, shape[i+1:], 1) for i in range(len(shape))]
  def process_shape(shape: tuple) -> list: return (lambda s: (s, eval("*".join(map(str, s))), len(s), (c_int * len(s))(*s)))(list(shape[0]) if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else list(shape))

class DtypeHelp:
  # dtype related helper functions
  dtype_map = {"float32": DType.FLOAT32, "float64": DType.FLOAT64, "int8": DType.INT8, "int16": DType.INT16, "int32": DType.INT32, "int64": DType.INT64, "uint8": DType.UINT8, "uint16": DType.UINT16, "uint32": DType.UINT32, "uint64": DType.UINT64, "bool": DType.BOOL}
  type_dtypes: list = ["int8", "int16", "int32", "int64", "long", "float32", "float64", "double", "uint8", "uint16", "uint32", "uint64", "bool"]

  def _parse_dtype(dtype:str) -> int: return DtypeHelp.dtype_map[dtype] if dtype in DtypeHelp.type_dtypes else (_ for _ in ()).throw(ValueError(f"Unsupported dtype: {dtype}. Supported dtypes: {DtypeHelp.type_dtypes}"))
  def get_dtypes() -> list: return DtypeHelp.type_dtypes

class Slice:
  def __init__(self, parent_tensor, row_index, shape, size, strides):
    self.parent_tensor, self.row_index, self.shape, self.size, self.strides, self.ndim = parent_tensor, row_index, shape, size, strides, len(shape)
  def __getitem__(self, sub_key):
    if isinstance(sub_key, int):
      if sub_key < 0: sub_key += self.shape[0]
      if sub_key < 0 or sub_key >= self.shape[0]: raise IndexError(f"Index {sub_key} out of bounds")
      if self.ndim == 1:
        indices = [self.row_index, sub_key]
        indices_ctypes = (c_int * len(indices))(*indices)
        return lib.get_item_tensor(self.parent_tensor.data, indices_ctypes)
      else: return Slice(self.parent_tensor, self.row_index, self.shape[1:], self.size // self.shape[0], self.strides[1:])
    elif isinstance(sub_key, tuple):
      indices = [self.row_index] + list(sub_key)
      if len(indices) > self.parent_tensor.ndim: raise IndexError(f"Too many indices")
      indices += [0] * (self.parent_tensor.ndim - len(indices))
      indices_ctypes = (c_int * len(indices))(*indices)
      return lib.get_item_tensor(self.parent_tensor.data, indices_ctypes)
    else: raise TypeError("Index must be int or tuple of ints")

  def __setitem__(self, sub_key, value):
    from .tensor import Tensor
    if isinstance(sub_key, int):
      if sub_key < 0: sub_key += self.shape[0]
      if sub_key < 0 or sub_key >= self.shape[0]: raise IndexError(f"Index {sub_key} out of bounds")
      indices = [self.row_index, sub_key]
      if len(indices) < self.parent_tensor.ndim: indices += [0] * (self.parent_tensor.ndim - len(indices))
      if isinstance(value, Tensor):
        if value.size != 1: raise ValueError("Can only assign scalar tensors")
        value = value.tolist()
      indices_ctypes = (c_int * len(indices))(*indices)
      lib.set_item_tensor(self.parent_tensor.data, indices_ctypes, c_float(value))
    elif isinstance(sub_key, tuple):
      indices = [self.row_index] + list(sub_key)
      if len(indices) > self.parent_tensor.ndim: raise IndexError(f"Too many indices")
      indices += [0] * (self.parent_tensor.ndim - len(indices))
      if isinstance(value, Tensor):
        if value.size != 1: raise ValueError("Can only assign scalar tensors")
        value = value.tolist()      
      indices_ctypes = (c_int * len(indices))(*indices)
      lib.set_item_tensor(self.parent_tensor.data, indices_ctypes, c_float(value))
    else: raise TypeError("Index must be int or tuple of ints")

def _set_item_tensor(self, key, value):
  from .tensor import Tensor
  if self.ndim == 0: raise TypeError("0-d tensor cannot be indexed")
  if isinstance(key, int):
    if key < 0: key += self.shape[0]
    if key < 0 or key >= self.shape[0]: raise IndexError(f"Index {key} out of bounds for dimension 0 with size {self.shape[0]}")

    if self.ndim == 1:
      indices = [key]
      if isinstance(value, Tensor):
        if value.size != 1: raise ValueError("Can only assign scalar tensors")
        value = value.tolist()
      indices_ctypes = (c_int * len(indices))(*indices)
      lib.set_item_tensor(self.data, indices_ctypes, c_float(value))
    else:
      if isinstance(value, (list, tuple)):
        flat_value = ShapeHelp.flatten(value)
        expected_size = self.size // self.shape[0]
        if len(flat_value) != expected_size: raise ValueError(f"Cannot assign {len(flat_value)} values to slice of size {expected_size}")

        for i, val in enumerate(flat_value):
          linear_idx = key * self.strides[0] + i
          temp_indices = []
          temp_linear = linear_idx
          for dim in range(self.ndim - 1, -1, -1):
            temp_indices.insert(0, temp_linear % self.shape[dim])
            temp_linear //= self.shape[dim]
          indices_ctypes = (c_int * len(temp_indices))(*temp_indices)
          lib.set_item_tensor(self.data, indices_ctypes, c_float(val))
      else: raise ValueError("Cannot assign scalar to multi-dimensional slice")
  elif isinstance(key, tuple):
    if len(key) > self.ndim: raise IndexError(f"Too many indices for tensor: got {len(key)}, expected {self.ndim}")
    indices = list(key) + [0] * (self.ndim - len(key))
    if isinstance(value, Tensor):
      if value.size != 1: raise ValueError("Can only assign scalar tensors")
      value = value.tolist()
    indices_ctypes = (c_int * len(indices))(*indices)
    lib.set_item_tensor(self.data, indices_ctypes, c_float(value))
  else: raise TypeError("Index must be int or tuple of ints")

def _iter_item_tensor(self):
  if self.ndim == 0: raise TypeError("Iteration over 0-d tensor")
  for i in range(self.shape[0]):
    if self.ndim == 1: yield self[i]
    else:
      row_data, offset, row_size = [], i * self.strides[0], self.size // self.shape[0]
      for j in range(row_size):
        linear_idx, indices = offset + j, []
        temp_idx = linear_idx
        for dim in range(self.ndim - 1, 0, -1):
          indices.insert(0, temp_idx % self.shape[dim])
          temp_idx //= self.shape[dim]
        indices.insert(0, i)
        indices_ctypes = (c_int * len(indices))(*indices)
        value = lib.get_item_tensor(self.data, indices_ctypes)
        row_data.append(value)
      yield ShapeHelp.reshape_list(row_data, self.shape[1:])

def _get_item_tensor(self, key):
  if self.ndim == 0: raise TypeError("0-d tensor cannot be indexed")
  if isinstance(key, int):
    if key < 0: key += self.shape[0]
    if key < 0 or key >= self.shape[0]: raise IndexError(f"Index {key} out of bounds for dimension 0 with size {self.shape[0]}")
    if self.ndim == 1:
      indices = [key]
      indices_ctypes = (c_int * len(indices))(*indices)
      return lib.get_item_tensor(self.data, indices_ctypes)
    else:
      new_shape = self.shape[1:]
      new_size = self.size // self.shape[0]
      new_strides = self.strides[1:]
      return Slice(self, key, new_shape, new_size, new_strides)
  elif isinstance(key, tuple):
    if len(key) > self.ndim: raise IndexError(f"Too many indices for tensor: got {len(key)}, expected {self.ndim}")
    indices = list(key) + [0] * (self.ndim - len(key))
    indices_ctypes = (c_int * len(indices))(*indices)
    return lib.get_item_tensor(self.data, indices_ctypes)
  else: raise TypeError("Index must be int or tuple of ints")