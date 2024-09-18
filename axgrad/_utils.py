""" here lies the functions to generate sample tensors """

from .helpers.utils import _zeros, _ones, _randint, _randn, _arange, _zeros_like, _ones_like
from ._dtype import Dtype
from typing import *

def zeros(shape:tuple, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None) -> list:
  return Dtype.handle_conversion(_zeros(shape), dtype if dtype is not None else 'float32')

def ones(shape:tuple, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None) -> list:
  return Dtype.handle_conversion(_ones(shape), dtype if dtype is not None else 'float32')

def randint(low:int, high:int, size:int=None, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None) -> list:
  return Dtype.handle_conversion(_randint(low, high, size), dtype if dtype is not None else 'float32')

def arange(start:int=0, end:int=10, step:int=1) -> list:
  return _arange(start, end, step)

def randn(domain:tuple=(1, -1), shape:tuple=None, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None) -> list:
  return Dtype.handle_conversion(_randn(domain, shape), dtype if dtype is not None else 'float32')

def zeros_like(arr:list, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None) -> list:
  return Dtype.handle_conversion(_zeros_like(arr), dtype if dtype is not None else 'float32')

def ones_like(arr:list, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None) -> list:
  return Dtype.handle_conversion(_ones_like(arr), dtype if dtype is not None else 'float32')