from typing import Any
from .helpers.shape import get_shape, flatten, _unsqueeze, broadcasted_shape, broadcast_array, _reshape
from .helpers.activations import gelu, tanh, sigmoid, relu
import math

class tensor:
  def __init__(self, *data, child:tuple=(), requires_grad:bool=False):
    self.data = data[0] if len(data) == 1 and isinstance(data[0], list) else list(data)
    self.shape = self.shape()
    self.ndim = len(self.shape)
    self.requires_grad = requires_grad
    self.grad = 0 if self.requires_grad else None
    self.prev = set(child)
    self.leaf = set()
    self._backward = lambda: None
    self.grad_fn = None

  def __repr__(self) -> str:
    data_str = ',\n\t'.join([str(row) for row in self.data])
    return f'tensor({data_str})'

  def __getitem__(self, index):
    return self.data[index]

  def __setattr__(self, name: str, value: Any) -> None:
    super().__setattr__(name, value)

  def __setitem__(self, index:int, value: Any):
    if isinstance(index, tuple):
      data = self.data
      grad = self.grad
      for idx in index[:-1]:
        data = data[idx]
        grad = grad[idx]
      data[index[-1]] = value
      grad[index[-1]] = value
    else:
      self.data[index] = value
      self.grad[index] = value

  def tolist(self):
    return self.data

  def zero_grad(self):
    self.grad = None

  def detach(self):
    self.grad = None
    self.grad_fn = None

  def _infer_shape(self, data):
    if isinstance(data, list):
      return [len(data)] + self._infer_shape(data[0])
    return []

  def shape(self):
    return get_shape(self.data)

  def size(self):
    return tuple(get_shape(self.data))

  def numel(self):
    out = 1
    for dim in self.shape:
      out = out * dim
    return out

  def flatten(self, start_dim:int=0, end_dim:int=-1):
    out = tensor(flatten(self.data, start_dim, end_dim), child=(self,))
    return out

  def unsqueeze(self, dim:int=0):
    out = tensor(_unsqueeze(self.data, dim), child=(self,))
    return out

  def squeeze(self):
    raise NotImplementedError("No function written!")

  def reshape(self, new_shape:tuple):
    reshaped = _reshape(self.data, new_shape)
    return tensor(reshaped, child=(self,))

  def broadcast(self, other):
    other = other if isinstance(other, tensor) else tensor(other)
    new_shape, needs_broadcasting = broadcasted_shape(self.shape, other.shape)

    if needs_broadcasting:
      other = tensor(broadcast_array(other.data, new_shape), child=(other,))
    return other

  def sum(self, axis:int=None, keepdim:bool=False):
    
    # doesn't work properly
    def _re_sum(data, axis):
      if axis is None:
        return [sum(flatten(data))]
      elif axis == 0:
        return [sum(row[i] for row in data) for i in range(len(data[0]))]
      else:
        for i in range(len(data[0])):
          for row in data:
            if isinstance(row[i], list):
              return _re_sum(row[i], axis-1)
            return [_re_sum(data[j], None) for j in range(len(data))]

    if axis is not None and (axis < 0 or axis >= len(self.shape)):
      raise ValueError("Axis out of range for the tensor")

    out = _re_sum(self.data, axis)
    if keepdim:
      if isinstance(out[0], list):
        out = [item for item in out]
    else:
      out = flatten(out)
    out = tensor(out, child=(self,))
    return out

  def __add__(self, other):
    other = other if isinstance(other, tensor) else tensor(other)
    def _add(a, b):
      if not isinstance(a, list):
        return a + b
      else:
        return [_add(ai, bi) for ai, bi in zip(a,b)]

    target_shape, requires_broadcasting = broadcasted_shape(self.shape, other.shape)

    if requires_broadcasting:
      self = tensor(broadcast_array(self.data, target_shape))
      other = tensor(broadcast_array(other.data, target_shape))

    if self.shape == other.shape:
      out = tensor(_add(self.data, other.data), child=(self, other))
    else:
      raise ValueError(f"invalid dimensions for addition. {self.shape} != {other.shape}")

    return out

  def __mul__(self, other):
    other = other if isinstance(other, tensor) else tensor(other)
    def _mul(a, b):
      if not isinstance(a, list):
        return a * b
      else:
        return [_mul(ai, bi) for ai, bi in zip(a,b)]

    target_shape, requires_broadcasting = broadcasted_shape(self.shape, other.shape)

    if requires_broadcasting:
      self = tensor(broadcast_array(self.data, target_shape))
      other = tensor(broadcast_array(other.data, target_shape))

    if self.shape == other.shape:
      out = tensor(_mul(self.data, other.data), child=(self, other))
    else:
      raise ValueError(f"invalid dimensions for addition. {self.shape} != {other.shape}")

    return out
  
  def __radd__(self, other):
    return other + self
  
  def __rmul__(self, other):
    return other * self
  
  def __truediv__(self, other):
    return self * other ** -1
  
  def __rtruediv__(self, other):
    return other * self ** -1
  
  def __sub__(self, other):
    return self + (-other)
  
  def __rsub__(self, other):
    return other + (-self)
  
  def __neg__(self):
    def _neg(data):
      if not isinstance(data, list):
        return -data
      return [_neg(sub_data) for sub_data in data]

    return tensor(_neg(self.data), child=(self,))
  
  def __pow__(self, pow):
    assert isinstance(pow, (int, float))
    
    def _pow(data, pow):
      if isinstance(data, list):
        return [_pow(sub_data, pow) for sub_data in data]
      return math.pow(data, pow)

    out = tensor(_pow(self.data, pow), child=(self,))
    return out
  
  def relu(self):
    def apply_activ(data):
      if isinstance(data, list):
        return [apply_activ(sub_data) for sub_data in data]
      else:
        return relu(data)
    out = tensor(apply_activ(self.data), child=(self,))
    return out

  def gelu(self):
    def apply_activ(data):
      if isinstance(data, list):
        return [apply_activ(sub_data) for sub_data in data]
      else:
        return gelu(data)
    out = tensor(apply_activ(self.data), child=(self,))
    return out

  def sigmoid(self):
    def apply_activ(data):
      if isinstance(data, list):
        return [apply_activ(sub_data) for sub_data in data]
      else:
        return sigmoid(data)
    out = tensor(apply_activ(self.data), child=(self,))
    return out

  def tanh(self):
    def apply_activ(data):
      if isinstance(data, list):
        return [apply_activ(sub_data) for sub_data in data]
      else:
        return tanh(data)
    out = tensor(apply_activ(self.data), child=(self,))
    return out