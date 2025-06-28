import math, functools
from ._core import DType
from ctypes import c_int

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