import math, functools
from ._core import DType
from ctypes import c_int

class HShape:
  # helper class for shape related functions
  get_shape:list = lambda data: [len(data)] + HShape.get_shape(data[0]) if isinstance(data, list) else []
  flatten:list = lambda subdata: [item for sub in subdata for item in HShape.flatten(sub)] if isinstance(subdata, list) else [subdata]
  get_strides:list = lambda shape: [] if not shape else [1] if len(shape) == 1 else [functools.reduce(lambda a, b: a * b, shape[i+1:], 1) for i in range(len(shape))]
  get_size:int = lambda shape: 1 if not shape else shape[0] * HShape.get_size(shape[1:])
  transposed_shape:list = lambda shape: shape if len(shape) == 1 else [shape[1], shape[0]] if len(shape) == 2 else [shape[0], shape[2], shape[1]] if len(shape) == 3 else (_ for _ in ()).throw(ValueError(f"Unsupported shape dimension: {len(shape)}"))
  reshape_list:list = lambda flat_list, shape: flat_list[:shape[0]] if len(shape) == 1 else [HShape.reshape_list(flat_list[i * (len(flat_list) // shape[0]):(i + 1) * (len(flat_list) // shape[0])], shape[1:]) for i in range(shape[0])]
  process_shape = lambda shape: (lambda s: (s, eval("*".join(map(str, s))), len(s), (c_int * len(s))(*s)))(list(shape[0]) if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else list(shape))

class HDtype:
  # helper class for dtype related functions
  type_dtypes = ["int8", "int16", "int32", "int64", "long", "float32", "float64", "double", "uint8", "uint16", "uint32", "uint64", "bool"]
  _parse_dtype = lambda dtype: {"float32": DType.FLOAT32, "float64": DType.FLOAT64, "int8": DType.INT8, "int16": DType.INT16, "int32": DType.INT32, "int64": DType.INT64, "uint8": DType.UINT8, "uint16": DType.UINT16, "uint32": DType.UINT32, "uint64": DType.UINT64, "bool": DType.BOOL}[dtype] if dtype in HDtype.type_dtypes else (_ for _ in ()).throw(ValueError(f"Unsupported dtype: {dtype}. Supported dtypes: {HDtype.type_dtypes}"))
  get_dtypes:list = lambda: HDtype.type_dtypes