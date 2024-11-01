"""
  @_backward.py entry file for autograd
  @brief Contains the backward logic implementation part for all the functions
  @comments:
  - just the initaiallization of the backwards
  - projecting & computing grads
"""

from typing import Literal, Callable
from ..helpers.functionals import *
from .functions.binary_ops import __ADD__, __MUL__, __MATMUL__, __POW__, __CONV2D__
from .functions.shape_ops import __TRANSPOSE__, __SWAPAXES__, __RESHAPE__, __SQUEEZE__, __UNSQUEEZE__, __FLATTEN__, __VIEW__, __BROADCAST__
from .functions.uniary_ops import __SUM__, __MEAN__, __VAR__, __STD__, __EXP__, __RSQRT__, __SQRT__, __LOG__, __CLIP__, __ABS__
from .functions.activations import __GELU__, __RELU__, __SIGMOID__, __SILU__, __TANH__, __LRELU__
from .functions.other_ops import __EMBEDD__, __STACK__, __CONCAT__, __BATCHNORM__, __LAYERNORM__
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
  
  def conv2d_backwards(first:Literal["tensor"], kernel:Union[Literal["tensor"], list], out:Literal["tensor"], stride:int, padding:int) -> Callable:
    _back = __CONV2D__(first, kernel, out, stride, padding)
    return _back

  ## unary ops backwards:
  def sum_backwards(out: Literal["tensor"], first: Literal["tensor"], axis: Optional[int], keepdims: bool) -> Callable:
    _back = __SUM__(first, out, axis, keepdims)
    return _back

  def mean_backwards(out: Literal["tensor"], first: Literal["tensor"], axis: Optional[int], keepdims: bool) -> Callable:
    _back = __MEAN__(first, out, axis, keepdims)
    return _back

  def var_backwards(out: Literal["tensor"], first: Literal["tensor"], axis: Optional[int], ddof:int, keepdims: bool) -> Callable:
    _back = __VAR__(first, out, axis, ddof, keepdims)
    return _back
  
  def std_backwards(out: Literal["tensor"], first: Literal["tensor"], axis: Optional[int], ddof:int, keepdims: bool) -> Callable:
    _back = __STD__(first, out, axis, ddof, keepdims)
    return _back
  
  def log_backwards(out:Literal["tensor"], first:Literal["tensor"]) -> Callable:
    _back = __LOG__(first, out)
    return _back
  
  def abs_backwards(out:Literal["tensor"], first:Literal["tensor"]) -> Callable:
    _back = __ABS__(first, out)
    return _back
  
  def exp_backwards(out:Literal["tensor"], first:Literal["tensor"]) -> Callable:
    _back = __EXP__(first, out)
    return _back
  
  def rsqrt_backwards(out:Literal["tensor"], first:Literal["tensor"]) -> Callable:
    _back = __RSQRT__(first, out)
    return _back
  
  def sqrt_backwards(out:Literal["tensor"], first:Literal["tensor"]) -> Callable:
    _back = __SQRT__(first, out)
    return _back
  
  ## other ops backwards:
  def embed_backwards(first:Literal["tensor"], indices:list, out:Literal["tensor"]) -> Callable:
    _back = __EMBEDD__(first, indices, out)
    return _back
  
  def stack_backwards(out:Literal["tensor"], tensors:Literal["tensor"], axis:int) -> Callable:
   _back = __STACK__(out, tensors, axis)
   return _back
  
  def concat_backwards(out:Literal["tensor"], tensors:Literal["tensor"], axis:int) -> Callable:
   _back = __CONCAT__(out, tensors, axis)
   return _back
  
  def layernorm_backwards(gamma:Literal["tensor"], beta:Literal["tensor"], bias:Literal["tensor"], out:Literal["tensor"], elem_aff:bool, eps:float, x:Literal["tensor"], mean:Literal["tensor"], var:Literal["tensor"]) -> Callable:
    _back = __LAYERNORM__(gamma, beta, bias, out, elem_aff, eps, x, mean, var)
    return _back
  
  def batchnorm_backwards(gamma:Literal["tensor"], beta:Literal["tensor"], running_mean:Literal["tensor"], running_var:Literal["tensor"] , aff:bool, out:Literal["tensor"], eps:float, x:Literal["tensor"], mean:Literal["tensor"], var:Literal["tensor"]) -> Callable:
    _back = __BATCHNORM__(gamma=gamma, beta=beta, running_mean=running_mean, running_var=running_var, out=out, affine=aff, eps=eps, x=x, mean=mean, var=var)
    return _back

  ## shape ops backwards:
  def transpose_backwards(first:Literal["tensor"], out:Literal["tensor"]) -> Callable:
    _back = __TRANSPOSE__(first, out)
    return _back

  def swapaxes_backwards(first, out, axis1, axis2) -> Callable:
    _back = __SWAPAXES__(first, out, axis1, axis2)
    return _back
  
  def reshape_backwards(first, out, new_shape) -> Callable:
    _back = __RESHAPE__(first, out, new_shape)
    return _back
  
  def view_backwards(first, out, original_shape) -> Callable:
    _back = __VIEW__(first, out, original_shape)
    return _back
  
  def unsqeeze_backwards(first, out, dim) -> Callable:
    _back = __UNSQUEEZE__(first, out, dim)
    return _back

  def sqeeze_backwards(first, out, dim) -> Callable:
    _back = __SQUEEZE__(first, out, dim)
    return _back
  
  def clip_backwards(out:Literal["tensor"], first:Literal["tensor"]) -> Callable:
    _back = __CLIP__(first, out)
    return _back

  def flatten_backwards(first, out, startdim, enddim) -> Callable:
    _back = __FLATTEN__(first, out, startdim, enddim)
    return _back
  
  def broadcast_backwards(first, out, new_shape) -> Callable:
    _back = __BROADCAST__(first, out, new_shape)
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