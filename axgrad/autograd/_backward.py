"""
  @_backward.py entry file for autograd
  @brief Contains the backward logic implementation part for all the functions
  @comments:
  - just the initaiallization of the backwards
  - projecting & computing grads
"""

from typing import List, Literal, Callable
from ..helpers.functionals import *
from .functions.binary_ops import __ADD__, __MUL__, __MATMUL__
from .functions.uniary_ops import __TRANSPOSE__, __SWAPAXES__, __RESHAPE__
from ..helpers.shape import *

class Backward:
  def add_backwards(out:Literal["tensor"], first:Literal["tensor"], second:Literal["tensor"]) -> Callable:
    _back = __ADD__(first, second, out)
    return _back
  
  def mul_backwards(out:Literal["tensor"], first:Literal["tensor"], second:Literal["tensor"]) -> Callable:
    _back = __MUL__(first, second, out)
    return _back

  def matmul_backwards(out:Literal["tensor"], first:Literal["tensor"], second:Literal["tensor"]) -> Callable:
    _back = __MATMUL__(first, second, out)
    return _back

  def transpose_backwards(first:Literal["tensor"], out:Literal["tensor"]) -> Callable:
    _back = __TRANSPOSE__(first, out)
    return _back

  def swapaxes_backwards(first, out, axis1, axis2) -> Callable:
    _back = __SWAPAXES__(first, out, axis1, axis2)
    return _back
  
  def reshape_backwards(frist, out, new_shape) -> Callable:
    _back = __RESHAPE__(frist, out, new_shape)
    return _back