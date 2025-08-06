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
  def is_broadcastable(shape_a:tuple, shape_b:tuple) -> bool:
    max_len, min_len = max(len(shape_a), len(shape_b)), min(len(shape_a), len(shape_b))
    longer, shorter = (shape_a, shape_b) if len(shape_a) >= len(shape_b) else (shape_b, shape_a)
    return all(s == l or s == 1 or l == 1 for s, l in zip(shorter[::-1], longer[max_len-min_len:][::-1]))
  def broadcast_shapes(shape_a:tuple, shape_b:tuple) -> tuple:
    max_len, longer, shorter = max(len(shape_a), len(shape_b)), (shape_a if len(shape_a) >= len(shape_b) else shape_b), (shape_b if len(shape_a) >= len(shape_b) else shape_a)
    padded_shorter, result = (1,) * (max_len - len(shorter)) + shorter, []
    for i in range(max_len): result.append(max(longer[i], padded_shorter[i]))
    return tuple(result), tuple(result)

class DtypeHelp:
  # dtype related helper functions
  dtype_map = {"float32": DType.FLOAT32, "float64": DType.FLOAT64, "int8": DType.INT8, "int16": DType.INT16, "int32": DType.INT32, "int64": DType.INT64, "uint8": DType.UINT8, "uint16": DType.UINT16, "uint32": DType.UINT32, "uint64": DType.UINT64, "bool": DType.BOOL}
  type_dtypes: list = ["int8", "int16", "int32", "int64", "long", "float32", "float64", "double", "uint8", "uint16", "uint32", "uint64", "bool"]

  def _parse_dtype(dtype:str) -> int: return DtypeHelp.dtype_map[dtype] if dtype in DtypeHelp.type_dtypes else (_ for _ in ()).throw(ValueError(f"Unsupported dtype: {dtype}. Supported dtypes: {DtypeHelp.type_dtypes}"))
  def get_dtypes() -> list: return DtypeHelp.type_dtypes

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
        tensor_list = value.tolist() if hasattr(value, 'tolist') else [value]
        value = float(tensor_list[0]) if isinstance(tensor_list, list) else float(tensor_list)
      elif isinstance(value, (list, tuple)):
        if len(value) != 1: raise ValueError("Can only assign single values to scalar positions")
        value = float(value[0])
      else: value = float(value)
      indices_ctypes = (c_int * len(indices))(*indices)
      lib.set_item_tensor(self.data, indices_ctypes, c_float(value))
    else:
      if isinstance(value, Tensor):
        if value.size != self.size // self.shape[0]: raise ValueError(f"Cannot assign tensor of size {value.size} to slice of size {self.size // self.shape[0]}")
        flat_value = value.tolist() if hasattr(value, 'tolist') else [value]
        flat_value = ShapeHelp.flatten(flat_value) if isinstance(flat_value, list) else [flat_value]
      elif isinstance(value, (list, tuple)):
        flat_value = ShapeHelp.flatten(value)
      else: raise ValueError("Cannot assign scalar to multi-dimensional slice")
      expected_size = self.size // self.shape[0]
      if len(flat_value) != expected_size: raise ValueError(f"Cannot assign {len(flat_value)} values to slice of size {expected_size}")
      for i, val in enumerate(flat_value):
        indices = [key]
        temp_size = expected_size
        for dim in range(1, self.ndim):
          dim_size = self.shape[dim]
          temp_size //= dim_size
          idx = (i // temp_size) % dim_size
          indices.append(idx)
        indices_ctypes = (c_int * len(indices))(*indices)
        lib.set_item_tensor(self.data, indices_ctypes, c_float(float(val)))

  elif isinstance(key, tuple):
    if len(key) > self.ndim: raise IndexError(f"Too many indices for tensor: got {len(key)}, expected {self.ndim}")
    indices = list(key) + [0] * (self.ndim - len(key))
    if isinstance(value, Tensor):
      if value.size != 1: raise ValueError("Can only assign scalar tensors")
      tensor_list = value.tolist() if hasattr(value, 'tolist') else [value]
      value = float(tensor_list[0]) if isinstance(tensor_list, list) else float(tensor_list)
    elif isinstance(value, (list, tuple)):
      if len(value) != 1: raise ValueError("Can only assign single values to indexed positions")
      value = float(value[0])
    else: value = float(value)
    indices_ctypes = (c_int * len(indices))(*indices)
    lib.set_item_tensor(self.data, indices_ctypes, c_float(value))
  else: raise TypeError("Index must be int or tuple of ints")

def _iter_item_tensor(self):
  if self.ndim == 0: raise TypeError("Iteration over 0-d tensor")
  for i in range(self.shape[0]):
    if self.ndim == 1:
      indices = [i]
      indices_ctypes = (c_int * len(indices))(*indices)
      yield lib.get_item_tensor(self.data, indices_ctypes)
    else: yield self[i]

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
      result = []
      slice_size = self.size // self.shape[0]
      def get_slice_recursive(base_indices, remaining_shape):
        if len(remaining_shape) == 1:
          row = []
          for j in range(remaining_shape[0]):
            indices = base_indices + [j]
            indices_ctypes = (c_int * len(indices))(*indices)
            row.append(lib.get_item_tensor(self.data, indices_ctypes))
          return row
        else:
          result = []
          for i in range(remaining_shape[0]):
            sub_result = get_slice_recursive(base_indices + [i], remaining_shape[1:])
            result.append(sub_result)
          return result
      return get_slice_recursive([key], self.shape[1:])
  elif isinstance(key, tuple):
    if len(key) > self.ndim: raise IndexError(f"Too many indices for tensor: got {len(key)}, expected {self.ndim}")
    indices = list(key) + [0] * (self.ndim - len(key))
    indices_ctypes = (c_int * len(indices))(*indices)
    return lib.get_item_tensor(self.data, indices_ctypes)
  else: raise TypeError("Index must be int or tuple of ints")