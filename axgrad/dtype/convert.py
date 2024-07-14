from .dtype import *

def convert_dtype(data, dtype):
  if dtype == 'int8':
    return [to_int8(val) for val in data]
  elif dtype == 'int16':
    return [to_int16(val) for val in data]
  elif dtype == 'int32':
    return [to_int32(val) for val in data]
  elif dtype == 'int64' or dtype == 'long':
    return [to_int64(val) for val in data]
  elif dtype == 'float16':
    return [to_float16(val) for val in data]
  elif dtype == 'float32':
    return [to_float32(val) for val in data]
  elif dtype == 'float64' or dtype == 'double':
    return [to_float64(val) for val in data]
  else:
    raise ValueError("Unsupported dtype")

def handle_conversion(data, dtype):
  if isinstance(data, list):
    return [handle_conversion(item, dtype) if isinstance(item, list) else convert_dtype([item], dtype)[0] for item in data]
  else:
    return convert_dtype([data], dtype)[0]