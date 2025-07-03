from ctypes import c_float, c_size_t, c_int, c_bool
from typing import *

from ._core import CTensor, lib, DType
from .helpers import ShapeHelp, DtypeHelp
from .autograd.functions import *

int8, int16, int32, int64, long = "int8", "int16", "int32", "int64", "long"
float32, float64, double = "float32", "float64", "double"
uint8, uint16, uint32, uint64 = "uint8", "uint16", "uint32", "uint64"
boolean = "bool"

class SumBackwards:
  def __init__(self, x, axis, keepdims): self.input = [x]; self.axis, self.keepdims = axis, keepdims
  def backward(self, grad):
    x = self.input[0]
    if self.axis == -1: return [grad.tolist()[0] * Tensor(lib.ones_like_tensor(x.data).contents, x.dtype, x.requires_grad)]
    else:
      if not self.keepdims: 
        new_shape = list(x.shape)
        new_shape[self.axis] = 1
        grad = grad.reshape(new_shape)
      return [grad * Tensor(lib.ones_like_tensor(x.data).contents, x.dtype, x.requires_grad)]

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
      if v in visited or not v.requires_grad: return
      visited.add(v)
      if v.grad_fn: [build_topo(inp) for inp in v.grad_fn.input if isinstance(inp, Tensor)]
      topo_order.append(v)

    build_topo(self)
    self.grad = gradient if self.grad is None else self.grad + gradient

    for tensor in reversed(topo_order):
      if tensor.grad_fn and tensor.grad is not None:
        grads = tensor.grad_fn.backward(tensor.grad)
        for inp, grad in zip(tensor.grad_fn.input, grads):
          if isinstance(inp, Tensor) and inp.requires_grad:
            inp.grad = grad if inp.grad is None else inp.grad + grad

  def __setattr__(self, name, value):
    if name == "grad": [setattr(self, "_temp_value", hook(getattr(self, "_temp_value", value))) for hook in self.hooks]; value = getattr(self, "_temp_value", value)
    super().__setattr__(name, value)
  def register_hook(self, function): self.hooks.append(function)
  def __str__(self): return (lib.print_tensor(self.data), "")[1]
  def astype(self, dtype: DType) -> "Tensor":
    out = Tensor(lib.cast_tensor(self.data, c_int(DtypeHelp._parse_dtype(dtype))).contents, requires_grad=self.requires_grad)
    out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
    return out
  def tolist(self) -> List[Any]:
    data_ptr = lib.out_data(self.data)
    data_tensor = [data_ptr[i] for i in range(self.size)]
    if self.ndim == 0: return data_tensor[0]
    elif self.ndim == 1: return data_tensor
    else: return ShapeHelp.reshape_list(data_tensor, self.shape)

  def transpose(self) -> "Tensor":
    assert self.ndim <= 3, ".transpose() ops limited to 3-d tensor"
    out = Tensor(lib.transpose_tensor(self.data).contents, self.dtype, self.requires_grad)
    out.shape, out.size, out.ndim = tuple(ShapeHelp.transpose_shape(self.shape)), self.size, self.ndim
    out.strides = ShapeHelp.get_strides(out.shape)
    if self.requires_grad: out.grad_fn = TransposeBackwards(self)
    return out

  def flatten(self) -> "Tensor":
    out = Tensor(lib.flatten_tensor(self.data).contents, self.dtype, self.requires_grad)
    out.shape, out.size, out.ndim = (self.size, ), self.size, 1
    out.strides = ShapeHelp.get_strides(out.shape)
    if self.requires_grad: out.grad_fn = FlatBackwards(self)
    return out

  def reshape(self, new_shape: Union[list, tuple]) -> "Tensor":
    if isinstance(new_shape, tuple): new_shape = list(new_shape)
    new_size, ndim = 1, len(new_shape)
    for dim in new_shape: new_size *= dim
    if new_size != self.size: raise ValueError(f"Cannot reshape Tensor of size {self.size} into shape {new_shape}")
    out = Tensor(lib.reshape_tensor(self.data, (c_int * ndim)(*new_shape), c_int(ndim)).contents, self.dtype, self.requires_grad)
    out.shape, out.size, out.ndim, out.strides = tuple(new_shape), self.size, ndim, ShapeHelp.get_strides(new_shape)
    if self.requires_grad: out.grad_fn = ReshapeBackwards(self)
    return out

  def sum(self, axis: int = -1, keepdims: bool = False) -> "Tensor":
    out = Tensor(lib.sum_tensor(self.data, c_int(axis), c_bool(keepdims)).contents, self.dtype, self.requires_grad)
    if axis == -1: out.shape, out.size, out.ndim = (1,) if keepdims else (), 1, 1 if keepdims else 0
    else:
      new_shape = list(self.shape)
      if keepdims: new_shape[axis] = 1
      else: new_shape.pop(axis)
      out.shape = tuple(new_shape)
      out.size, out.ndim, out.strides = 1 if not new_shape else eval('*'.join(map(str, new_shape))), len(new_shape), ShapeHelp.get_strides(out.shape) if out.shape else []
    if self.requires_grad: out.grad_fn = SumBackwards(self, axis, keepdims)
    return out

  def __add__(self, other) -> "Tensor":
    other_tensor = other if isinstance(other, Tensor) else Tensor([other] if isinstance(other, (int, float)) else other, self.dtype)
    result_ptr = lib.add_scalar_tensor(self.data, c_float(other)).contents if isinstance(other, (int, float)) else lib.add_tensor(self.data, other_tensor.data).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad or (isinstance(other, Tensor) and other.requires_grad))
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if out.requires_grad: out.grad_fn = AddBackwards(self, other_tensor if isinstance(other, Tensor) else other)
    return out

  def __sub__(self, other) -> "Tensor":
    other_tensor = other if isinstance(other, Tensor) else Tensor([other] if isinstance(other, (int, float)) else other, self.dtype)
    result_ptr = lib.sub_scalar_tensor(self.data, c_float(other)).contents if isinstance(other, (int, float)) else lib.sub_tensor(self.data, other_tensor.data).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad or (isinstance(other, Tensor) and other.requires_grad))
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if out.requires_grad: out.grad_fn = SubBackwards(self, other_tensor if isinstance(other, Tensor) else other)
    return out

  def __mul__(self, other) -> "Tensor":
    other_tensor = other if isinstance(other, Tensor) else Tensor([other] if isinstance(other, (int, float)) else other, self.dtype)
    result_ptr = lib.mul_scalar_tensor(self.data, c_float(other)).contents if isinstance(other, (int, float)) else lib.mul_tensor(self.data, other_tensor.data).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad or (isinstance(other, Tensor) and other.requires_grad))
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if out.requires_grad: out.grad_fn = MulBackwards(self, other_tensor if isinstance(other, Tensor) else other)
    return out

  def __truediv__(self, other) -> "Tensor":
    other_tensor = other if isinstance(other, Tensor) else Tensor([other] if isinstance(other, (int, float)) else other, self.dtype)
    result_ptr = lib.div_scalar_tensor(self.data, c_float(other)).contents if isinstance(other, (int, float)) else lib.div_tensor(self.data, other_tensor.data).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad or (isinstance(other, Tensor) and other.requires_grad))
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if out.requires_grad: out.grad_fn = DivBackwards(self, other_tensor if isinstance(other, Tensor) else other)
    return out

  def __matmul__(self, other):
    other = other if isinstance(other, (CTensor, Tensor)) else Tensor(other, self.dtype)
    if self.ndim <= 2 and other.ndim <= 2:
      result_ptr = lib.matmul_tensor(self.data, other.data).contents
    elif self.ndim == 3 and other.ndim == 3 and self.shape[0] == other.shape[0]:
      result_ptr = lib.batch_matmul_tensor(self.data, other.data).contents
    else:
      result_ptr = lib.broadcasted_matmul_tensor(self.data, other.data).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad or other.requires_grad)
    shape, ndim, size = lib.out_shape(out.data), self.ndim, lib.out_size(out.data)
    out.shape, out.ndim, out.size = tuple([shape[i] for i in range(ndim)]), ndim, size
    out.strides = ShapeHelp.get_strides(out.shape)
    if out.requires_grad: out.grad_fn = MatmulBackwards(self, other)
    return out

  def __neg__(self) -> "Tensor":
    result_pointer = lib.neg_tensor(self.data).contents
    out = Tensor(result_pointer, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if out.requires_grad: out.grad_fn = NegBackwards(self)
    return (setattr(out, "grad", self.grad), setattr(out, "hooks", self.hooks), setattr(out, "grad_fn", self.grad_fn), out)[3] if self.requires_grad else out

  def __radd__(self, other): return self + other
  def __rsub__(self, other): return Tensor([other], self.dtype, self.requires_grad) - self
  def __rmul__(self, other): return self * other
  def __rtruediv__(self, other): return Tensor([other], self.dtype, self.requires_grad) / self

  def __pow__(self, exp) -> "Tensor":
    if isinstance(exp, (int, float)): result_ptr = lib.pow_tensor(self.data, c_float(exp)).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = PowBackwards(self, exp)
    return out

  def __rpow__(self, base) -> "Tensor":
    if isinstance(base, (int, float)): result_ptr = lib.pow_scalar(c_float(base), self.data).contents
    else: raise NotImplementedError("__rpow__ with Tensor base not implemented yet")
    out = Tensor(result_ptr, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = RPowBackwards(base, self)
    return out

  def sign(self) -> "Tensor":
    result_pointer = lib.sign_tensor(self.data).contents
    out = Tensor(result_pointer, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    return (setattr(out, "grad", self.grad), setattr(out, "hooks", self.hooks), setattr(out, "grad_fn", self.grad_fn), out)[3] if self.requires_grad else out

  def log(self) -> "Tensor":
    result_pointer = lib.log_tensor(self.data).contents
    out = Tensor(result_pointer, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = LogBackwards(self)
    return out

  def exp(self) -> "Tensor":
    result_pointer = lib.exp_tensor(self.data).contents
    out = Tensor(result_pointer, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = ExpBackwards(self)
    return out

  def abs(self) -> "Tensor":
    result_pointer = lib.abs_tensor(self.data).contents
    out = Tensor(result_pointer, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = AbsBackwards(self)
    return out

  def sqrt(self) -> "Tensor":
    result_pointer = lib.sqrt_tensor(self.data).contents
    out = Tensor(result_pointer, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = SqrtBackwards(self)
    return out

  def sin(self) -> "Tensor":
    result_ptr = lib.sin_tensor(self.data).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = SinBackwards(self)
    return out

  def cos(self) -> "Tensor":
    result_ptr = lib.cos_tensor(self.data).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = CosBackwards(self)
    return out

  def tan(self) -> "Tensor":
    result_ptr = lib.tan_tensor(self.data).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = TanBackwards(self)
    return out

  def sinh(self) -> "Tensor":
    result_ptr = lib.sinh_tensor(self.data).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = SinhBackwards(self)
    return out

  def cosh(self) -> "Tensor":
    result_ptr = lib.cosh_tensor(self.data).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = CoshBackwards(self)
    return out

  def tanh(self) -> "Tensor":
    result_ptr = lib.tanh_tensor(self.data).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = TanhBackwards(self, out)
    return out

  def sigmoid(self) -> "Tensor":
    result_ptr = lib.sigmoid_tensor(self.data).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = SigmoidBackwards(self, out)
    return out

  def relu(self) -> "Tensor":
    result_ptr = lib.relu_tensor(self.data).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = ReluBackwards(self, out)
    return out

  def elu(self, alpha:float = 1e-5) -> "Tensor":
    out = Tensor(lib.elu_tensor(self.data, c_float(alpha)).contents, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = EluBackwards(self, out, alpha)  # Passing alpha parameter
    return out

  def leaky_relu(self, eps:float = 1e-5) -> "Tensor":
    out = Tensor(lib.leaky_relu_tensor(self.data, c_float(eps)).contents, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = LeakyReluBackwards(self, out, eps)  # Passing eps parameter
    return out

  def swish(self, beta:float = 0.5) -> "Tensor":
    out = Tensor(lib.swish_tensor(self.data, c_float(beta)).contents, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = SwishBackwards(self, out, beta)  # Passing beta parameter
    return out

  def silu(self) -> "Tensor":
    result_ptr = lib.silu_tensor(self.data).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = SiluBackwards(self, out)
    return out

  def softplus(self) -> "Tensor":
    result_ptr = lib.softplus_tensor(self.data).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = SoftplusBackwards(self, out)
    return out

  def gelu(self) -> "Tensor":
    result_ptr = lib.gelu_tensor(self.data).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = GeluBackwards(self, out)
    return out

  def dot(self, other):
    other = other if isinstance(other, (CTensor, Tensor)) else Tensor(other, self.dtype)
    if self.ndim <= 2 and other.ndim <= 2:
      result_ptr = lib.dot_tensor(self.data, other.data).contents
    elif self.ndim == 3 and other.ndim == 3 and self.shape[0] == other.shape[0]:
      result_ptr = lib.batch_dot_tensor(self.data, other.data).contents
    out = Tensor(result_ptr, self.dtype, self.requires_grad or other.requires_grad)
    shape, ndim, size = lib.out_shape(result_ptr), out.data.ndim, lib.out_size(result_ptr)
    out.shape, out.ndim, out.size = tuple([shape[i] for i in range(ndim)]), ndim, size
    out.strides = ShapeHelp.get_strides(out.shape)
    if out.requires_grad: out.grad_fn = DotBackwards(self, other)
    return out