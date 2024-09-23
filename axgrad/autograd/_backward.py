"""
  @_backward.py entry file for autograd
  @brief Contains the backward logic implementation part for all the functions
  @comments:
  - just the initaiallization of the backwards
  - projecting & computing grads
"""

from typing import Literal, Callable
from ..helpers.functionals import *
from .functions.binary_ops import __ADD__, __MUL__, __MATMUL__, __POW__
from .functions.uniary_ops import __TRANSPOSE__, __SWAPAXES__, __RESHAPE__
from .functions.activations import __GELU__, __RELU__, __SIGMOID__, __SILU__, __TANH__, __LRELU__
from ..helpers.shape import *

class Backward:
  ## binary ops backwards:
  def add_backwards(out:Literal["tensor"], first:Literal["tensor"], second:Literal["tensor"]) -> Callable:
    _back = __ADD__(first, second, out)
    return _back
  
  def mul_backwards(out:Literal["tensor"], first:Literal["tensor"], second:Literal["tensor"]) -> Callable:
    _back = __MUL__(first, second, out)
    return _back
  
  def pow_backwards(out:Literal["tensor"], first:Literal["tensor"], power:Union[int, float]) -> Callable:
    _back = __POW__(first, out, power)
    return _back

  def matmul_backwards(out:Literal["tensor"], first:Literal["tensor"], second:Literal["tensor"]) -> Callable:
    _back = __MATMUL__(first, second, out)
    return _back

  ## unary ops backwards:
  def transpose_backwards(first:Literal["tensor"], out:Literal["tensor"]) -> Callable:
    _back = __TRANSPOSE__(first, out)
    return _back

  def swapaxes_backwards(first, out, axis1, axis2) -> Callable:
    _back = __SWAPAXES__(first, out, axis1, axis2)
    return _back
  
  def reshape_backwards(first, out, new_shape) -> Callable:
    _back = __RESHAPE__(first, out, new_shape)
    return _back
  
  ## activation functions backwards:
  def relu_backwards(first, out) -> Callable:
    _back = __RELU__(first, out)
    return _back
  
  def gelu_backwards(first, out) -> Callable:
    _back = __GELU__(first, out)
    return _back
  
  def tanh_backwards(first, out) -> Callable:
    _back = __TANH__(first, out)
    return _back

  def sigmoid_backwards(first, out) -> Callable:
    _back = __SIGMOID__(first, out)
    return _back
  
  def lrelu_backwards(first, out) -> Callable:
    _back = __LRELU__(first, out)
    return _back
  
  def silu_backwards(first, out) -> Callable:
    _back = __SILU__(first, out)
    return _back