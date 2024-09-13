"""
  @dtype_str/convert.py
  @brief Contains code to change dtype of every element in a tensor
"""

# from dataclasses import dataclass
import struct

def to_int8(value):
  return struct.unpack('b', struct.pack('b', int(value)))[0]

def to_int16(value):
  return struct.unpack('h', struct.pack('h', int(value)))[0]

def to_int32(value):
  return struct.unpack('i', struct.pack('i', int(value)))[0]

def to_int64(value):
  return struct.unpack('q', struct.pack('q', int(value)))[0]

def to_float16(value):
  return struct.unpack('e', struct.pack('e', float(value)))[0]

def to_float32(value):
  return struct.unpack('f', struct.pack('f', float(value)))[0]

def to_float64(value):
  return struct.unpack('d', struct.pack('d', float(value)))[0]

# @dataclass(frozen=True, order=True)
class Dtype:
  @staticmethod
  def convert_dtype(data, dtype_str):
    if dtype_str == "int8":
      return [to_int8(val) for val in data]
    elif dtype_str == "int16":
      return [to_int16(val) for val in data]
    elif dtype_str == "int32":
      return [to_int32(val) for val in data]
    elif dtype_str == "int64" or dtype_str == "long":
      return [to_int64(val) for val in data]
    elif dtype_str == "float16":
      return [to_float16(val) for val in data]
    elif dtype_str == "float32":
      return [to_float32(val) for val in data]
    elif dtype_str == "float64" or dtype_str == "double":
      return [to_float64(val) for val in data]
    else:
      raise ValueError(f"Unsupported dtype: {dtype_str}")

  @staticmethod
  def handle_conversion(data, dtype_str):
    if isinstance(data, list):
      return [Dtype.handle_conversion(item, dtype_str) if isinstance(item, list) else Dtype.convert_dtype([item], dtype_str)[0] for item in data]
    else:
      return Dtype.convert_dtype([data], dtype_str)[0]