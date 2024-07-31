from typing import *
from .helpers.shape import *
from .dtypes.convert import handle_conversion
from .helpers.utils import *
from copy import deepcopy
from .helpers.functionals import *

int8 = 'int8'
int16 = 'int16'
int32 = 'int32'
int64 = 'int64'
long = 'long'
float16 = 'float16'
float32 = 'float32'
float64 = 'float64'
double = 'double'

class tensor:
  int8 = int8
  int16 = int16
  int32 = int32
  int64 = int64 or long
  float16 = float16
  float32 = float32
  float64 = float64 or double

  def __init__(self, *data, requires_grad:bool=True, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None) -> None:
    if len(data) == 1 and isinstance(data[0], list):
      self.data = data[0]
    else:
      list(data)
    self.dtype = tensor.int32 if dtype is None else dtype
    if dtype is not None:
      self.data = handle_conversion(self.data, dtype)
    self.shape = self.shape()
    self.size = tuple(self.shape)
    self.ndim = len(self.shape)
    self.prev = set()
    self.grad_fn = None
    self.grad = None
    self.requires_grad = requires_grad
    if self.requires_grad is True:
      self.grad = zeros(self.shape)
    self._backward = lambda: None
  
  def __repr__(self) -> str:
    if self.ndim == 1:
      return f'tensor([{self.data}], dtype={self.dtype})'
    data_str = ',\n\t'.join([str(row) for row in self.data])
    return f'tensor([{data_str}], dtype={self.dtype})'

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

  def astype(self, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]) -> List['tensor']:
    new_data = handle_conversion(self.data, dtype)
    return tensor(new_data, requires_grad=self.requires_grad)

  def tolist(self) -> list:
    return self.data
  
  def copy(self) -> List['tensor']:
    return tensor(deepcopy(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
  
  def shape(self) -> list:
    return get_shape(self.data)
  
  def numel(self) -> int:
    out = 1
    for dim in self.shape:
      out *= dim
    return out

  def view(self, dtype:Optional[Literal['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]=None) -> List['tensor']:
    new_array = tensor(self.data, requires_grad=self.requires_grad)
    if dtype is not None:
      new_array.data = handle_conversion(new_array.data, dtype)
      new_array.dtype = dtype
    return new_array
  
  def detach(self) -> None:
    self.grad = None
    self.grad_fn = None
  
  def zero_grad(self) -> None:
    self.grad = None

  # unary operations -------------------
  
  def flatten(self, start_dim:int=0, end_dim:int=-1) -> List['tensor']:
    out = tensor(flatten_recursive(self.data, start_dim, end_dim), requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev = set(self,)
    return out
  
  def F(self) -> List['tensor']:
    out = tensor(flatten(self.data), requires_grad=self.requires_grad, dtype=self.data)
    out.prev = set(self,)
  
  def transpose(self, dim) -> List['tensor']:
    out = tensor(transpose_recursive(self.data, dim), requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev = set(self,)
    return out
  
  def T(self) -> List['tensor']:
    out = tensor(transpose(self.data), requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev = set(self,)
    return out
  
  def reshape(self, new_shape:tuple) -> List['tensor']:
    out = tensor(reshape(self.data), requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev = set(self,)
    return out
  
  def broadcast(self, other) -> List['tensor']:
    if isinstance(self, tensor):
      other = other
    else:
      other = tensor(other, requires_grad=self.requires_grad, dtype=self.dtype)  
    new_shape, needs_broad = broadcast_shape(self.shape, other.shape)
    if needs_broad:
      other = tensor(broadcast(other.data, new_shape), requires_grad=self.requires_grad, dtype=self.dtype)
      other.prev = other.prev
      if other.requires_grad:
        other.grad = broadcast(other.grad, new_shape)
    return other
  
  def sum(self, axis=None, keepdims=False) -> List['tensor']:
    if axis is not None and (axis >= self.ndim):
      raise ValueError("Axis out of range for the tensor")
    
    axis = axis + self.ndim if axis < 0 else axis
    out = tensor(tensor_sum(self.data, axis, keepdims), requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev = set(self, )
    return out
  
  def mean(self, axis:Optional[int]=None, keepdims:bool=False) -> List['tensor']:
    if axis is None:
      flat_array = self.F()
      mean_val = sum(flat_array) / len(flat_array)
      if keepdims:
        return [[mean_val]]
      out = mean_val
    else:
      out = mean_axis(self.data, axis, keepdims)
    out = tensor(out, requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev = set(self, )
    return out

  def var(self, axis:Optional[int]=None, ddof:int=0, keepdims:bool=False) -> List['tensor']:
    if axis is None:
      flat_array = flatten(self.data)
      mean_value = sum(flat_array) / len(flat_array)
      variance = sum((x - mean_value) ** 2 for x in flat_array) / (len(flat_array) - ddof)
      if keepdims:
        return [[variance]]
      out = variance
    else:
      mean_values = self.mean(axis=axis)
      out = var_axis(self.data, mean_values, axis, ddof, keepdims)
    out = tensor(out, requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev = set(self, )
    return out

  def std(self, axis:Optional[int]=None, ddof:int=0, keepdims:bool=False) -> List['tensor']:
    variance = self.var(axis=axis, ddof=ddof, keepdims=keepdims)
    def _std(var):
      if isinstance(var, list):
        return [_std(sub) for sub in var]
      return math.sqrt(var)
    if keepdims:
      out = [[math.sqrt(x)] for x in flatten(variance)]
    else:
      out = _std(variance)
    out = tensor(out, requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev = set(out)
    return out
  
  def unsqueeze(self, dim:int=0) -> List['tensor']:
    out = tensor(unsqueeze(self.data, dim), requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev =set(out, )
    return out
  
  def squeeze(self, dim:int=0) -> List['tensor']:
    if dim is not None and (dim<0 or dim>=self.ndim):
      raise IndexError(f"Dimension out of range (expected to be in range of {self.ndim} dimensions)")

    out = tensor(squeeze(self.data, dim), requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev =set(out, )
    return out
  
  def clip(self, min_value, max_value) -> List['tensor']:
    def _clip(data, min_value, max_value):
      if isinstance(data, list):
        return [_clip(d, min_value, max_value) for d in data]
      return max(min(data, max_value), min_value)
    
    out = tensor(_clip(self.data, min_value, max_value))
    out.prev = set(self, )
    return out
  
  # binary operations -------------------
  
  def __add__(self, other) -> List['tensor']:
    if isinstance(self, tensor):
      other = other
    else:
      other = tensor(other, requires_grad=self.requires_grad, dtype=self.dtype)
    
    def _ops(a, b):
      if isinstance(a, list):
        return [_ops(_a, _b) for _a, _b in zip(a, b)]
      else:
        return a + b
    
    target_shape, req_broad = broadcast_shape(self.shape, other.shape)
    if req_broad:
      self = tensor(broadcast(self, target_shape), requires_grad=self.requires_grad, dtype=self.dtype)
      other = tensor(broadcast(other, target_shape), requires_grad=other.requires_grad, dtype=other.dtype)

      if self.requires_grad:
        self.grad = broadcast(self.grad, target_shape)
        other.grad = broadcast(other.grad, target_shape)
    
    if self.shape == other.shape:
      out = tensor(_ops(self.data, other.data), requires_grad=self.requires_grad, dtype=self.dtype)
      out.prev = set(self, other)
    else:
      raise ValueError("shapes are incompatible for operation")
    return out
  
  def __mul__(self, other) -> List['tensor']:
    if isinstance(self, tensor):
      other = other
    else:
      other = tensor(other, requires_grad=self.requires_grad, dtype=self.dtype)
    
    def _ops(a, b):
      if isinstance(a, list):
        return [_ops(_a, _b) for _a, _b in zip(a, b)]
      else:
        return a * b
    
    target_shape, req_broad = broadcast_shape(self.shape, other.shape)
    if req_broad:
      self = tensor(broadcast(self, target_shape), requires_grad=self.requires_grad, dtype=self.dtype)
      other = tensor(broadcast(other, target_shape), requires_grad=other.requires_grad, dtype=other.dtype)

      if self.requires_grad:
        self.grad = broadcast(self.grad, target_shape)
        other.grad = broadcast(other.grad, target_shape)
    
    if self.shape == other.shape:
      out = tensor(_ops(self.data, other.data), requires_grad=self.requires_grad, dtype=self.dtype)
      out.prev = set(self, other)
    else:
      raise ValueError("shapes are incompatible for operation")
    return out
  
  def __rmul__(self, other) -> List['tensor']:
    return other * self

  def __neg__(self) -> List['tensor']:
    def _ops(a):
      if isinstance(a, list):
        return [_ops(_a) for _a in a]
      else:
        return -a
    out = tensor(_ops(self.data), requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev = self.prev
    return out

  def __sub__(self, other) -> List['tensor']:
    return self + (-other)

  def __rsub__(self, other) -> List['tensor']:
    return other + (-self)
  
  def __pow__(self, pow:Union[int, float], eps:float=1e6) -> List['tensor']:
    assert isinstance(pow, (int, float)), "power exponent is of incompatible datatype"

    def _ops(data, pow):
      if isinstance(data, list):
        return [_ops(_d, pow) for _d in data]
      if data == 0:
        data = eps
      return math.pow(data, pow)
    
    out = tensor(_ops(self.data), requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev = set(self, )
    return out
  
  def __truediv__(self, other) -> List['tensor']:
    return self * (other ** -1)
  
  def __rtruediv__(self, other) -> List['tensor']:
    return other * (self ** -1)
  
  def relu(self) -> List['tensor']:
    def _ops(data):
      if isinstance(data, list):
        return [_ops(_d) for _d in data]
      else:
        return relu(data)
    out = tensor(self.data, requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev = set(self, )
    return out
  
  def gelu(self) -> List['tensor']:
    def _ops(data):
      if isinstance(data, list):
        return [_ops(_d) for _d in data]
      else:
        return gelu(data)
    out = tensor(self.data, requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev = set(self, )
    return out

  def tanh(self) -> List['tensor']:
    def _ops(data):
      if isinstance(data, list):
        return [_ops(_d) for _d in data]
      else:
        return tanh(data)
    out = tensor(self.data, requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev = set(self, )
    return out

  def sigmoid(self) -> List['tensor']:
    def _ops(data):
      if isinstance(data, list):
        return [_ops(_d) for _d in data]
      else:
        return sigmoid(data)
    out = tensor(self.data, requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev = set(self, )
    return out

  def silu(self) -> List['tensor']:
    def _ops(data):
      if isinstance(data, list):
        return [_ops(_d) for _d in data]
      else:
        return silu(data)
    out = tensor(self.data, requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev = set(self, )
    return out