from ctypes import c_float, c_size_t, c_int
from typing import *

from ._core import CTensor, lib, DType
from .helpers import ShapeHelp, DtypeHelp, _set_item_tensor, _iter_item_tensor, _get_item_tensor
from .autograd.functions import *
from .ops.binary import *
from .ops.functional import *
from .ops.unary import log_tensor_ops, sign_tensor_ops, sqrt_tensor_ops, abs_tensor_ops, exp_tensor_ops, neg_tensor_ops
from .ops.shape import flatten_tensor_ops, transpose_tensor_ops, reshape_tensor_ops, squeeze_tensor_ops, unsqueeze_tensor_ops, contiguous_tensor_ops, view_tensor_ops, make_contiguous_tensor_ops
from .ops.redux import sum_tensor_ops, var_tensor_ops, mean_tensor_ops, std_tensor_ops, max_tensor_ops, min_tensor_ops
from .ops.norm import clip_tensor_ops, clamp_tensor_ops, register_norm_ops

int8, int16, int32, int64, long = "int8", "int16", "int32", "int64", "long"
float32, float64, double = "float32", "float64", "double"
uint8, uint16, uint32, uint64 = "uint8", "uint16", "uint32", "uint64"
boolean = "bool"

class Tensor:
  int8, int16, int32, int64, long, float32, float64, double, uint8, uint16, uint32, uint64, boolean = int8, int16, int32, int64, long, float32, float64, double, uint8, uint16, uint32, uint64, boolean  
  def __init__(self, data: Union[List[Any], int, float], dtype: str=float32, requires_grad: bool=False):
    if isinstance(data, CTensor): self.data, self.shape, self.size, self.ndim, self.strides, self.dtype = data, (), 0, 0, [], dtype or "float32"
    elif isinstance(data, Tensor): self.data, self.shape, self.dtype, self.size, self.ndim, self.strides = data.data, data.shape, dtype or data.dtype, data.size, data.ndim, data.strides
    else:
      data, shape = ShapeHelp.flatten([data] if isinstance(data, (int, float)) else data), tuple(ShapeHelp.get_shape(data))
      self.size, self.ndim, self.dtype, self.shape, self.strides = len(data), len(shape), dtype or "float32", shape, ShapeHelp.get_strides(shape)
      self._data_ctypes, self._shape_ctypes = (c_float * self.size)(*data.copy()), (c_int * self.ndim)(*shape)
      self.data = lib.create_tensor(self._data_ctypes, c_size_t(self.ndim), self._shape_ctypes, c_size_t(self.size), c_int(DtypeHelp._parse_dtype(self.dtype)))
    self.requires_grad, self.hooks, self.grad_fn, self.grad = requires_grad, [], None, None

  def backward(self, gradient=None):
    assert self.ndim == 0 or (self.ndim == 1 and self.size == 1), "backward can only be called for scalar tensors"
    if gradient is None: gradient = Tensor([1.0], dtype=self.dtype)
    visited, topo_order = set(), []
    def build_topo(v):
      if id(v) in visited or not v.requires_grad: return
      visited.add(id(v))
      if v.grad_fn: [build_topo(inp) for inp in v.grad_fn.input if isinstance(inp, Tensor)]
      topo_order.append(v)
    build_topo(self)
    self.grad = gradient if self.grad is None else self.grad + gradient
    for tensor in reversed(topo_order):
      if tensor.grad_fn and tensor.grad is not None:
        grads = tensor.grad_fn.backward(tensor.grad)
        for inp, grad in zip(tensor.grad_fn.input, grads):
          if isinstance(inp, Tensor) and inp.requires_grad: inp.grad = grad if inp.grad is None else inp.grad + grad

  def __setattr__(self, name, value):
    if name == "grad": [setattr(self, "_temp_value", hook(getattr(self, "_temp_value", value))) for hook in self.hooks]; value = getattr(self, "_temp_value", value)
    super().__setattr__(name, value)

  def register_hook(self, function): self.hooks.append(function)
  def __str__(self): return (lib.print_tensor(self.data), "")[1]
  def __hash__(self): return id(self)
  def __getitem__(self, key): return _get_item_tensor(self, key)
  def __setitem__(self, key, value): _set_item_tensor(self, key, value)
  def __iter__(self): return _iter_item_tensor(self)
  def contiguous(self): return contiguous_tensor_ops(self)
  def make_contiguous(self): return make_contiguous_tensor_ops(self)
  def view(self): return view_tensor_ops(self)

  def astype(self, dtype: DType) -> "Tensor":
    out = Tensor(lib.cast_tensor(self.data, c_int(DtypeHelp._parse_dtype(dtype))).contents, requires_grad=self.requires_grad)
    out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
    return out

  def tolist(self):
    if self.size == 0: return []
    if self.ndim == 0: 
      indices_ctypes = (c_int * 1)(0)
      return lib.get_item_tensor(self.data, indices_ctypes)
    elif self.ndim == 1: return [self[i] for i in range(self.shape[0])]
    else: return [self[i] for i in range(self.shape[0])]

  def __eq__(self, other):
    if isinstance(other, (int, float)):
      out = Tensor(lib.equal_scalar(self.data, c_float(other)).contents, DType.BOOL, self.requires_grad)
    else:
      other = other if isinstance(other, Tensor) else Tensor(other, self.dtype, False)
      out = Tensor(lib.equal_tensor(self.data, other.data).contents, DType.BOOL, self.requires_grad)
    return (setattr(out, "grad", self.grad), setattr(out, "hooks", self.hooks), setattr(out, "grad_fn", self.grad_fn), out)[3] if self.requires_grad else out

  def __ne__(self, other):
    if isinstance(other, (int, float)):
      out = Tensor(lib.not_equal_scalar(self.data, c_float(other)).contents, DType.BOOL, self.requires_grad)
    else:
      other = other if isinstance(other, Tensor) else Tensor(other, self.dtype, False)
      out = Tensor(lib.equal_tensor(self.data, other.data).contents, DType.BOOL, self.requires_grad)
    return (setattr(out, "grad", self.grad), setattr(out, "hooks", self.hooks), setattr(out, "grad_fn", self.grad_fn), out)[3] if self.requires_grad else out

  def __gt__(self, other):
    if isinstance(other, (int, float)):
      out = Tensor(lib.greater_scalar(self.data, c_float(other)).contents, DType.BOOL, self.requires_grad)
    else:
      other = other if isinstance(other, Tensor) else Tensor(other, self.dtype, False)
      out = Tensor(lib.greater_tensor(self.data, other.data).contents, DType.BOOL, self.requires_grad)
    return (setattr(out, "grad", self.grad), setattr(out, "hooks", self.hooks), setattr(out, "grad_fn", self.grad_fn), out)[3] if self.requires_grad else out

  def __ge__(self, other):
    if isinstance(other, (int, float)):
      out = Tensor(lib.greater_equal_scalar(self.data, c_float(other)).contents, DType.BOOL, self.requires_grad)
    else:
      other = other if isinstance(other, Tensor) else Tensor(other, self.dtype, False)
      out = Tensor(lib.greater_equal_tensor(self.data, other.data).contents, DType.BOOL, self.requires_grad)
    return (setattr(out, "grad", self.grad), setattr(out, "hooks", self.hooks), setattr(out, "grad_fn", self.grad_fn), out)[3] if self.requires_grad else out

  def __lt__(self, other):
    if isinstance(other, (int, float)):
      out = Tensor(lib.smaller_scalar(self.data, c_float(other)).contents, DType.BOOL, self.requires_grad)
    else:
      other = other if isinstance(other, Tensor) else Tensor(other, self.dtype, False)
      out = Tensor(lib.smaller_tensor(self.data, other.data).contents, DType.BOOL, self.requires_grad)
    return (setattr(out, "grad", self.grad), setattr(out, "hooks", self.hooks), setattr(out, "grad_fn", self.grad_fn), out)[3] if self.requires_grad else out

  def __le__(self, other):
    if isinstance(other, (int, float)):
      out = Tensor(lib.smaller_equal_scalar(self.data, c_float(other)).contents, DType.BOOL, self.requires_grad)
    else:
      other = other if isinstance(other, Tensor) else Tensor(other, self.dtype, False)
      out = Tensor(lib.smaller_equal_tensor(self.data, other.data).contents, DType.BOOL, self.requires_grad)
    return (setattr(out, "grad", self.grad), setattr(out, "hooks", self.hooks), setattr(out, "grad_fn", self.grad_fn), out)[3] if self.requires_grad else out

  def __add__(self, other): return add_tensor_ops(self, other)
  def __radd__(self, other): return radd_tensor_ops(self, other)
  def __sub__(self, other): return sub_tensor_ops(self, other)
  def __rsub__(self, other): return rsub_tensor_ops(self, other)
  def __mul__(self, other): return mul_tensor_ops(self, other)
  def __rmul__(self, other): return rmul_tensor_ops(self, other)
  def __truediv__(self, other): return div_tensor_ops(self, other)
  def __rtruediv__(self, other): return rdiv_tensor_ops(self, other)
  def __pow__(self, exp: Union[float, int]): return pow_tensor_ops(self, exp)
  def __rpow__(self, base: 'Tensor'): return rpow_tensor_ops(self, base)
  def __matmul__(self, other): return matmul_tensor_ops(self, other)
  def __neg__(self): return neg_tensor_ops(self)
  def log(self): return log_tensor_ops(self)
  def exp(self): return exp_tensor_ops(self)
  def sign(self): return sign_tensor_ops(self)
  def abs(self): return abs_tensor_ops(self)
  def sqrt(self): return sqrt_tensor_ops(self)
  def sin(self): return sin_tensor_ops(self)
  def cos(self): return cos_tensor_ops(self)
  def tan(self): return tan_tensor_ops(self)
  def sinh(self): return sinh_tensor_ops(self)
  def cosh(self): return cosh_tensor_ops(self)
  def tanh(self): return tanh_tensor_ops(self)
  def sigmoid(self): return sigmoid_tensor_ops(self)
  def relu(self): return relu_tensor_ops(self)
  def gelu(self): return gelu_tensor_ops(self)
  def elu(self, alpha: float=1e-5): return elu_tensor_ops(self, alpha)
  def leakyrelu(self, eps: float=1e-5): return leaky_relu_tensor_ops(self, eps)
  def swish(self, beta: float=0.5): return swish_tensor_ops(self, beta)
  def silu(self): return silu_tensor_ops(self)
  def softplus(self): return softplus_tensor_ops(self)
  def sum(self, axis: int=-1, keepdims: bool=False): return sum_tensor_ops(self, axis, keepdims)
  def mean(self, axis: int=-1, keepdims: bool=False): return mean_tensor_ops(self, axis, keepdims)
  def var(self, axis: int=-1, ddof: int=0): return var_tensor_ops(self, axis, ddof)
  def std(self, axis: int=-1, ddof: int=0): return std_tensor_ops(self, axis, ddof)
  def max(self, axis: int=-1, keepdims: bool=False): return max_tensor_ops(self, axis, keepdims)
  def min(self, axis: int=-1, keepdims: bool=False): return min_tensor_ops(self, axis, keepdims)
  def transpose(self): return transpose_tensor_ops(self)
  def flatten(self): return flatten_tensor_ops(self)
  def reshape(self, new_shape: Union[Tuple, List]): return reshape_tensor_ops(self, new_shape)
  def clip(self, max_val: float): return clip_tensor_ops(self, max_val)
  def clamp(self, min_val: float,  max_val: float): return clamp_tensor_ops(self, min_val, max_val)
  def squeeze(self, axis: int): return squeeze_tensor_ops(self, axis)
  def unsqueeze(self, axis: int): return unsqueeze_tensor_ops(self, axis)

register_norm_ops()