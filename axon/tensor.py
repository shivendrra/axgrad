from .helpers.shape import get_shape, broadcast_array, broadcast_shapes, _flatten, _squeeze, _unsqueeze, _reshape
from .helpers.statics import zeros, ones
from .helpers.acitvations import relu, sigmoid, tanh, gelu
from .backward import backward
import math

class tensor:
  def __init__(self, *data, child:tuple=(), _ops:str=''):
    self.data = data[0] if len(data) == 1 and isinstance(data[0], list) else list(data)
    self.shape = self.shape()
    self.grad = zeros(self.shape, dtype=float)
    self.prev = set(child)
    self.leaf = set()
    self._backward = lambda: None
    self._ops = None if _ops == '' else _ops

  def __repr__(self):
    data_str = '\n\t'.join([str(row) for row in self.data])
    return f'axon.tensor(data={data_str}'
  
  def __getitem__(self, index):
    return self.data[index]
  
  def __setitem__(self, index, value):
    if isinstance(index, tuple):
      data_ref = self.data
      grad_ref = self.grad
      for idx in index[:-1]:
        data_ref = data_ref[idx]
        grad_ref = grad_ref[idx]
      data_ref[index[-1]] = value
      grad_ref[index[-1]] = value
    else:
      self.data[index] = value
      self.grad[index] = value
  
  def __add__(self, other):
    other = other if isinstance(other, tensor) else tensor(other)
    if self.shape != other.shape:
      raise ValueError(f'Arrays must be of same shape & size {self.shape} != {other.shape}')
    
    def _add(x, y):
      if not isinstance(x, list):
        return x + y
      return [_add(xi, yi) for xi, yi in zip(x, y)]
    
    out = tensor(_add(self.data, other.data), child=(self, other), _ops='<ElemLevelAdd>')
    out._backward = backward.add_backward(self, other, out)
    return out
  
  def __mul__(self, other):
    other = other if isinstance(other, tensor) else tensor(other)
    if self.shape != other.shape:
      raise ValueError(f'Arrays must be of same shape & size {self.shape} != {other.shape}')
    
    def _mul(x, y):
      if not isinstance(x, list):
        return x * y
      return [_mul(xi, yi) for xi, yi in zip(x, y)]
    out = tensor(_mul(self.data, other.data), child=(self, other), _ops='<ElemLevelMul>')      
    out._backward = backward.mul_backward(self, other, out)
    return out
  
  def __sub__(self, other):
    return self + (-other)
  
  def __truediv__(self, other):
    return self * other ** -1

  def __neg__(self):
    def _neg(data):
      if not isinstance(data, list):
        return -data
      return [_neg(sub_data) for sub_data in data]

    return _neg(self.data)

  def __rsub__(self, other):
    return other + (-self)

  def __rmul__(self, other):
    return self * other
  
  def __rtruediv__(self, other):
    return other * self**-1

  def __pow__(self, pow):
    assert isinstance(pow, (int, float))

    def apply_pow(data, pow):
      if isinstance(data, list):
        return [apply_pow(sub_data, pow) for sub_data in data]
      else:
        return math.pow(data, pow)

    out = apply_pow(self.data, pow)
    return tensor(out, child=(self,), _ops='<pow>')

  def relu(self):
    def apply_relu(data):
      if isinstance(data, list):
        return [apply_relu(sub_data) for sub_data in data]
      else:
        return relu(data)
    
    out =  tensor(apply_relu(self.data), child=(self,), _ops='<relu>')
    out._backward = backward.relu_backward(self, out)
    return out

  def tanh(self):
    def apply_tanh(data):
      if isinstance(data, list):
        return [apply_tanh(sub_data) for sub_data in data]
      else:
        return tanh(data)
    out = tensor(apply_tanh(self.data), child=(self,), _ops='<tanh>')
    out._backward = backward.tanh_backward(self, out)
    return out
  
  def gelu(self):
    def apply_gelu(data):
      if isinstance(data, list):
        return [apply_gelu(sub_data) for sub_data in data]
      else:
        return gelu(data)
    out = apply_gelu(self.data)
    return tensor(out, child=(self,), _ops='<gelu>')

  def sigmoid(self):
    def apply_sigmoid(data):
      if isinstance(data, list):
        return [apply_sigmoid(sub_data) for sub_data in data]
      else:
        return sigmoid(data)
    out = apply_sigmoid(self.data)
    return tensor(out, child=(self,), _ops='<sigmoid>')

  def backward(self):
    topo = backward.backward(self)
    self.grad = ones(self.shape, dtype=float)
    self.leaf = topo
    for node in reversed(topo):
      node._backward()

  def shape(self):
    return get_shape(self.data)
  
  def transpose(self):
    rows = len(self.data)
    cols = len(self.data[0])
    return tensor([[self.data[i][j] for i in range(rows)] for j in range(cols)])
  
  def flatten(self):
    new = _flatten(self.data)
    return new
  
  def squeeze(self, dim:int=None):
    if dim is not None and (dim<0 or dim>=len(self.shape)):
      raise IndexError(f'Dimension out of range (expected to be in range of {len(self.shape)} dimensions)')
    
    s_data = _squeeze(self.data, dim)
    return tensor(s_data, child=(self,), _ops='<squeeze>')

  def unsqeeze(self, dim:int):
    if dim < 0 or dim > len(self.shape):
      raise IndexError(f'Dimension out of range (expected to be in range of {len(self.shape) + 1} dimensions)')

    u_data = _unsqueeze(self.data, dim)
    return tensor(u_data, child=(self,), _ops='<unsqueeze>')

  def reshape(self, new_shape:tuple):
    flat_data = self.flatten()
    reshaped = _reshape(self.data, flat_data, new_shape)
    return tensor(reshaped, child=(self,), _ops='<reshape>')

  def sum(self, dtype=None):
    unpacked_arr = _flatten(self.data)
    out = sum(i for i in unpacked_arr)
    return dtype(out) if dtype is not None else out
  
  def broadcast(self, other):
    other = other if isinstance(other, tensor) else tensor(other)
    new_shape = broadcast_shapes(self.shape, other.shape)
    other_broadcasted = broadcast_array(other.data, new_shape)
    return tensor(other_broadcasted)