from typing import *
from .helpers.shape import *
from .dtypes.convert import handle_conversion
from .helpers.utils import _zeros, _ones
from .helpers.ops import *
from copy import deepcopy
from .helpers.functionals import *

int8 = "int8"
int16 = "int16"
int32 = "int32"
int64 = "int64"
long = "long"
float16 = "float16"
float32 = "float32"
float64 = "float64"
double = "double"

class tensor:
  int8 = int8
  int16 = int16
  int32 = int32
  int64 = int64 or long
  float16 = float16
  float32 = float32
  float64 = float64 or double

  def __init__(self, *data, dtype:Optional[Literal["int8", "int16", "int32", "int64", "float16", "float32", "float64"]]=None, requires_grad:bool=True) -> None:
    if len(data) == 1 and isinstance(data[0], list):
      data = data[0]
    else:
      data = list(data)
    self.dtype = tensor.int32 if dtype is None else dtype
    self.data = handle_conversion(data, self.dtype)
    self.shape = self.shape()
    self.grad_fn = None
    self.requires_grad = requires_grad
    self.prev = set()
    if self.requires_grad is True:
      self.grad = _zeros(self.shape)
    else:
      self.grad = None
    self._backward = lambda: None
  
  def __repr__(self) -> str:
    return f"tensor([{self.data}])"

  def __str__(self) -> str:
    def format_element(element):
      if isinstance(element, list):
        return [format_element(sub_element) for sub_element in element]
      if self.dtype == int8 or self.dtype == int16 or self.dtype == int32 or self.dtype == int64 or self.dtype == long:
        return f"{element:.0f}."
      if self.dtype == float16:
        return f"{element:.2f}"
      if self.dtype == float32:
        return f"{element:.3f}"
      return f"{element:.4f}"

    formatted_data = format_element(self.data)

    def format_data(data, level=0):
      if isinstance(data[0], list):
        inner = ",\n".join(["\t" * (level + 1) + format_data(sub_data, level + 1) for sub_data in data])
        return f"[\n{inner}\n" + "  " * level + "]"
      return "[" + ", ".join(data) + "]"

    formatted_str = format_data(formatted_data, 0)
    formatted_str = formatted_str.replace("\t", " ")
    return f"tensor({formatted_str}, grad_fn={self.grad_fn})\n"  if self.requires_grad else f"tensor({formatted_str}, dtype={self.dtype})\n"

  def __getitem__(self, index:tuple):
    if isinstance(index, tuple):
      data = self.data
      for idx in index[:-1]:
        data = data[idx]
      return data[index[-1]]
    else:
      return self.data[index]
  
  def __setattr__(self, name: str, value: Any) -> None:
    super().__setattr__(name, value)
  
  def __setitem__(self, index:tuple, value: Any) -> None:
    if isinstance(index, tuple):
      data = self.data
      for idx in index[:-1]:
        data = data[idx]
      data[index[-1]] = value
    else:
      self.data[index] = value

  def __iter__(self) -> Iterator:
    for item in self.data:
      yield item

  def astype(self, dtype:Optional[Literal["int8", "int16", "int32", "int64", "float16", "float32", "float64"]]) -> List["tensor"]:
    new_data = handle_conversion(self.data, dtype)
    out = tensor(new_data, dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = self.prev
    out.grad = self.grad
    out.grad_fn = self.grad_fn
    return out

  def tolist(self) -> list:
    return self.data
  
  def copy(self) -> List["tensor"]:
    out = tensor(deepcopy(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = self.prev
    out.grad = self.grad
    out.grad_fn = self.grad_fn
    return out
  
  def shape(self) -> list:
    return get_shape(self.data)

  @property
  def ndim(self):
    return len(self.shape)
  
  @property
  def size(self):
    return tuple(self.shape)
  
  @property
  def numel(self) -> int:
    out = 1
    for dim in self.shape:
      out *= dim
    return out

  @property
  def T(self):
    out = tensor(transpose(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<TransposeBackwards>"
    return out
  
  @property
  def F(self):
    out = tensor(flatten(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<FlattenBackwards>"
    return out

  def view(self, dtype:Optional[Literal["int8", "int16", "int32", "int64", "float16", "float32", "float64"]]=None) -> List["tensor"]:
    new_array = tensor(self.data, requires_grad=self.requires_grad)
    if dtype is not None:
      new_array.data = handle_conversion(new_array.data, dtype)
      new_array.dtype = dtype
    return new_array
  
  def detach(self) -> None:
    self.grad = None
    self.grad_fn = None
  
  def zero_grad(self) -> None:
    self.grad = _zeros(self.shape)

  # unary operations -------------------

  def swap_axes(self, axis1:int, axis2:int) -> List["tensor"]:
    axis1 = self.ndim + axis1 if axis1 < 0 else axis1
    axis2 = self.ndim + axis2 if axis2 < 0 else axis2
    out = tensor(swap_axes(self.data, axis1, axis2), dtype=self.dtype)
    out.prev = (self, )
    out.grad_fn = "<TransposeBackwards>"
    return out

  def unsqueeze(self, dim:int=0):
    dim = dim if dim > 0 else self.ndim + dim
    out = tensor(unsqueeze(self.data, dim), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<UnsqueezeBackwards>"
    return out
  
  def squeeze(self, dim:int=0):
    if dim is not None and dim>=self.ndim:
      raise IndexError(f"Dimension out of range (expected to be in range of {self.ndim} dimensions)")
    dim = dim if dim > 0 else self.ndim + dim
    out = tensor(squeeze(self.data, dim), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<SqueezeBackwards>"
    return out
  
  def reshape(self, new_shape:tuple) -> List["tensor"]:
    out = reshape(self.data, new_shape)
    out = tensor(out, dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<ReshapeBackwards>"
    return out

  def clip(self, min_value, max_value):
    def _clip(data, min_value, max_value):
      if isinstance(data, list):
        return [_clip(d, min_value, max_value) for d in data]
      return max(min(data, max_value), min_value)
    
    return tensor(_clip(self.data, min_value, max_value))

  def flatten(self, start_dim:int=0, end_dim:int=-1) -> List["tensor"]:
    out = tensor(flatten_recursive(self.data, start_dim, end_dim), requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev = set(self,)
    out.grad_fn = "<FlattenBackwards>"
    return out
  
  def transpose(self, dim) -> List["tensor"]:
    out = tensor(transpose_recursive(self.data, dim), requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev = set(self,)
    out.grad_fn = "<TransposeBackwards>"
    return out
  
  def sum(self, axis:Optional[int]=None, keepdims:bool=False):
    if axis == None:
      if keepdims:
        out = [[sum(flatten(self.data))]]
      else:
        out = sum(flatten(self.data))
    elif axis == 0:
      out = sum_axis0(self.data)
    else:
      out = sum_axis(self.data, axis, keepdims)
    out = tensor(out, dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<SumBackwards>"
    return out
  
  def broadcast(self, other:List["tensor"]) -> List["tensor"]:
    other = other if isinstance(other, tensor) else tensor(other, dtype=self.dtype, requires_grad=self.requires_grad)
    new_shape, needs_broadcasting = broadcast_shape(self.shape, other.shape)
    if needs_broadcasting:
      out = tensor(broadcast(other.data, new_shape), dtype=self.dtype, requires_grad=self.requires_grad)
      out.prev = (self, )
      out.grad_fn = "<BroadcastBackwards>"
      return out
    else:
      return None
  
  def dot(self, other:List["tensor"]) -> List["tensor"]:
    out = dot_product(self.data, other.data)
    out = tensor(out, dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<DotBackward>"
    return out
  
  def det(self) -> List["tensor"]:
    out = determinant(self.data)
    out = tensor(out, dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<DetBackward>"
    return out

  def mean(self, axis:Optional[int]=None, keepdims:bool=False) -> List["tensor"]:
    if axis is None:
      flat_array = flatten(self.data)
      mean_val = sum(flat_array) / len(flat_array)
      if keepdims:
        out = [[mean_val]]
      return mean_val
    if axis == 0:
      out = mean_axis0(self.data)
    else:
      out = mean_axis(self.data, axis, keepdims)
    out = tensor(out, dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<MeanBackwards>"
    return out

  def var(self, axis:Optional[int]=None, ddof:int=0, keepdims:bool=False) -> List["tensor"]:
    if axis is None:
      flat_array = flatten(self.data)
      mean_value = sum(flat_array) / len(flat_array)
      variance = sum((x - mean_value) ** 2 for x in flat_array) / (len(flat_array) - ddof)
      if keepdims:
        out = [[variance]]
      return variance
    if axis == 0:
      out = var_axis0(self.data)
    else:
      mean_values = self.mean(axis=axis)
      out = var_axis(self.data, mean_values, axis, ddof, keepdims)
    out = tensor(out, dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<VarBackward>"
    return out

  def std(self, axis:Optional[int]=None, ddof:int=0, keepdims:bool=False) -> list[float]:
    variance = self.var(axis=axis, ddof=ddof, keepdims=keepdims).data
    def _std(var):
      if isinstance(var, list):
        return [_std(sub) for sub in var]
      return math.sqrt(var)
    if keepdims:
      out = [[math.sqrt(x)] for x in flatten(variance)]
    else:
      out = _std(variance)
    out = tensor(out, dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<StdBackward>"
    return out

  # binary operations -------------------
  
  def __add__(self, other) -> List["tensor"]:
    if isinstance(other, tensor):
      other = other
    else:
      other = tensor(other, requires_grad=self.requires_grad, dtype=self.dtype)
    
    def _ops(a, b):
      if isinstance(a, list):
        return [_ops(_a, _b) for _a, _b in zip(a, b)]
      else:
        return a + b

    target_shape, requires_broadcasting = broadcast_shape(self.shape, other.shape)
    
    if requires_broadcasting:
      self.data = handle_conversion(broadcast(self.data, target_shape), self.dtype)
      self.shape = get_shape(self.data)
      other.data = handle_conversion(broadcast(other.data, target_shape), other.dtype)
      other.shape = get_shape(other.data)
    
    if self.size == other.size:
      out = tensor(_ops(self.data, other.data), dtype=self.dtype, requires_grad=self.requires_grad)
      out.prev = (self, other)
      out.grad_fn = "<AddBackward>"
    else:
      raise ValueError("shapes are incompatible for operation")
    return out
  
  def __mul__(self, other) -> List["tensor"]:
    if isinstance(other, tensor):
      other = other
    else:
      other = tensor(other, requires_grad=self.requires_grad, dtype=self.dtype)
    
    def _ops(a, b):
      if isinstance(a, list):
        return [_ops(_a, _b) for _a, _b in zip(a, b)]
      else:
        return a * b
    
    target_shape, requires_broadcasting = broadcast_shape(self.shape, other.shape)

    if requires_broadcasting:
      self.data = handle_conversion(broadcast(self.data, target_shape), self.dtype)
      self.shape = get_shape(self.data)
      other.data = handle_conversion(broadcast(other.data, target_shape), other.dtype)
      other.shape = get_shape(other.data)
    
    if self.size == other.size:
      out = tensor(_ops(self.data, other.data), dtype=self.dtype, requires_grad=self.requires_grad)
      out.prev = (self, other)
      out.grad_fn = "<MulBackward>"
    else:
      raise ValueError("shapes are incompatible for operation")
    return out
  
  def __rmul__(self, other) -> List["tensor"]:
    return other * self

  def __matmul__(self, other) -> List["tensor"]:
    if isinstance(other, tensor):
      other = other
    else:
      other = tensor(other, requires_grad=self.requires_grad, dtype=self.dtype)

    out = tensor(matmul(self.data, other.data), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, other)
    out.grad_fn = "<MatmulBackward>"
    return out

  def __neg__(self) -> List["tensor"]:
    def _ops(a):
      if isinstance(a, list):
        return [_ops(_a) for _a in a]
      else:
        return -a
    out = tensor(_ops(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = self.prev
    out.grad_fn = "<NegBackward>"
    return out

  def __sub__(self, other) -> List["tensor"]:
    return self + (-other)

  def __rsub__(self, other) -> List["tensor"]:
    return other + (-self)
  
  def __pow__(self, pow:Union[int, float], eps:float=1e6) -> List["tensor"]:
    assert isinstance(pow, (int, float)), "power exponent is of incompatible datatype"

    def _ops(data, pow):
      if isinstance(data, list):
        return [_ops(_d, pow) for _d in data]
      if data == 0:
        data = eps
      return math.pow(data, pow)
    
    out = tensor(_ops(self.data, pow), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<PowBackward>"
    return out
  
  def __truediv__(self, other) -> List["tensor"]:
    return self * (other ** -1)
  
  def __rtruediv__(self, other) -> List["tensor"]:
    return other * (self ** -1)
  
  def relu(self) -> List["tensor"]:
    def _ops(data):
      if isinstance(data, list):
        return [_ops(_d) for _d in data]
      else:
        return relu(data)
    out = tensor(_ops(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<ReluBackward>"
    return out
  
  def gelu(self) -> List["tensor"]:
    def _ops(data):
      if isinstance(data, list):
        return [_ops(_d) for _d in data]
      else:
        return gelu(data)
    out = tensor(_ops(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<GeluBackward>"
    return out

  def tanh(self) -> List["tensor"]:
    def _ops(data):
      if isinstance(data, list):
        return [_ops(_d) for _d in data]
      else:
        return tanh(data)
    out = tensor(_ops(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<TanhBackward>"
    return out

  def sigmoid(self) -> List["tensor"]:
    def _ops(data):
      if isinstance(data, list):
        return [_ops(_d) for _d in data]
      else:
        return sigmoid(data)
    out = tensor(_ops(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<SigmoidBackward>"
    return out

  def silu(self) -> List["tensor"]:
    def _ops(data):
      if isinstance(data, list):
        return [_ops(_d) for _d in data]
      else:
        return silu(data)
    out = tensor(_ops(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
    out.prev = (self, )
    out.grad_fn = "<SiluBackward>"
    return out