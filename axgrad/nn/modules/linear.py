from ...tensor import Tensor
from ..parameter import Parameter
from ..module import Module
from ..._core import lib
from ...helpers import ShapeHelp, DtypeHelp
from ...autograd.functions import *

from ctypes import c_int, c_size_t, c_float
import math

def _tensor_matmul(x: Tensor, y: Tensor, dtype: str= "float32") -> Tensor:
  result_data = lib.matmul_tensor(x.data, y.data).contents

  if x.ndim == 1 and y.ndim == 2: out_shape = (y.shape[1],)
  elif x.ndim == 2 and y.ndim == 2: out_shape = (x.shape[0], y.shape[1])
  else: raise ValueError(f"Unsupported matmul dimensions: {x.shape} @ {y.shape}")

  out = Tensor(result_data, dtype, requires_grad=x.requires_grad or y.requires_grad)
  out.shape, out.size, out.ndim = out_shape, 1, len(out_shape)
  for dim in out_shape: out.size *= dim
  out.strides = ShapeHelp.get_strides(out.shape)
  if out.requires_grad: out.grad_fn = MatmulBackwards(x, y)
  return out

class Linear(Module):
  def __init__(self, _in: int, _out: int, bias: bool = False, dtype: str= "float32"):
    self._in, self._out, self.bias, self.dtype = _in, _out, bias, dtype

    # Initialize weight parameter with Xavier/Glorot uniform initialization
    # std = sqrt(6 / (_in + _out))
    std = math.sqrt(6.0 / (_in + _out))
    self.weight = Parameter((_out, _in), dtype)
    self.weight.set_name("weight")
    self._init_uniform_weights(self.weight, -std, std)
    self.use_bias = False
    if bias:
      self.bias = Parameter((_out,), dtype)
      self.bias.set_name("bias")
      self.use_bias = True
      self._init_zero_weights(self.bias)
    else: self.bias = None
  
  def _init_uniform_weights(self, param: Parameter, low: float, high: float):
    uniform_data = lib.uniform_tensor(c_int(int(low * 10)), c_int(int(high * 10)), (c_int * param.ndim)(*param.shape), c_size_t(param.size),
      c_size_t(param.ndim), c_int(DtypeHelp._parse_dtype(param.dtype))).contents
    param.data = uniform_data

  def _init_zero_weights(self, param: Parameter):
    zeros_data = lib.zeros_tensor((c_int * param.ndim)(*param.shape), c_size_t(param.size), c_size_t(param.ndim), c_int(DtypeHelp._parse_dtype(param.dtype))).contents
    param.data = zeros_data

  def forward(self, x: Tensor) -> Tensor:
    assert x.shape[-1] == self._in, f"Input feature size {x.shape[-1]} doesn't match layer input size {self._in}"
    # Perform matrix multiplication: x @ weight^T
    # weight is (_out, _in), so we need weight^T which is (_in, _out)
    weight_t = self.weight.transpose()

    if x.ndim == 1 or x.ndim == 2: out = _tensor_matmul(x, weight_t, self.dtype) # Single sample: (_in,) @ (_in, _out) = (_out,)
    else:
      original_shape, batch_size = x.shape, 1
      for dim in original_shape[:-1]: batch_size *= dim

      x_reshaped = x.reshape((batch_size, self._in))  # Reshape to (batch_size, _in)
      out = _tensor_matmul(x_reshaped, weight_t, self.dtype)
      new_shape = original_shape[:-1] + (self._out,)    # Reshape back to (..., _out)
      out = out.reshape(new_shape)
    if self.use_bias: out + self.bias
    return out

  def __call__(self, x: Tensor) -> Tensor: return self.forward(x)
  def __repr__(self) -> str: return f"Linear(_in={self._in}, _out={self._out}, bias={self.use_bias})"
  def inner_repr(self) -> str: return f"_in={self._in}, _out={self._out}, bias={self.use_bias}"
  def parameters(self):
    yield self, "weight", self.weight
    if self.bias is not None:
      yield self, "bias", self.bias

  def zero_grad(self):
    for param in self.parameters(): param.zero_grad()