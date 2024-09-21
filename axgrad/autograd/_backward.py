"""
  @_backward.py entry file for autograd
  @brief Contains the backward logic implementation part for all the functions
  @comments:
  - just the initaiallization of the backwards
  - projecting & computing grads
"""

from typing import List, Literal
from ..helpers.functionals import *
from .functions.binary_ops import __ADD__, __MUL__, __MATMUL__
from ..helpers.shape import *

class Backward:
  def add_backwards(out:Literal["tensor"], first:Literal["tensor"], second:Literal["tensor"]):
    _back = __ADD__(first, second, out)
    return _back
  
  def mul_backwards(out:Literal["tensor"], first:Literal["tensor"], second:Literal["tensor"]):
    _back = __MUL__(first, second, out)
    return _back

  def matmul_backwards(out:Literal["tensor"], first:Literal["tensor"], second:Literal["tensor"]):
    _back = __MATMUL__(first, second, out)
    return _back

  def transpose_backwards(out, first):
    out.grad = transpose(first.grad)
    return out.grad

  def swapaxes_backwards(out, first, axis1, axis2):
    out.grad = swap_axes(first.grad, axis1, axis2)