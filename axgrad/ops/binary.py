from .main import add_tensor, sub_tensor, mul_tensor, div_tensor
from ..helpers.shape import broadcast, broadcast_shape

def _ensure_tensor(obj, ref):
  from .._core import _tensor
  return obj if isinstance(obj, _tensor) else _tensor(obj, ref.dtype)

def add_tensor_ops(self, other):
  from .._core import _tensor
  other = _ensure_tensor(other, self)
  target_shape, _ = broadcast_shape(self.shape, other.shape)
  self_data, other_data = broadcast(self.data, target_shape), broadcast(other.data, target_shape)
  out = add_tensor(self_data, other_data)
  return _tensor(out, self.dtype)

def radd_tensor_ops(self, other):
  # __radd__ is same as __add__ with operands flipped
  return add_tensor_ops(other, self)

def sub_tensor_ops(self, other):
  from .._core import _tensor
  other = _ensure_tensor(other, self)
  target_shape, _ = broadcast_shape(self.shape, other.shape)
  self_data, other_data = broadcast(self.data, target_shape), broadcast(other.data, target_shape)
  out = sub_tensor(self_data, other_data)
  return _tensor(out, self.dtype)

def rsub_tensor_ops(self, other):
  return sub_tensor_ops(other, self)

def mul_tensor_ops(self, other):
  from .._core import _tensor
  other = _ensure_tensor(other, self)
  target_shape, _ = broadcast_shape(self.shape, other.shape)
  self_data, other_data = broadcast(self.data, target_shape), broadcast(other.data, target_shape)
  out = mul_tensor(self_data, other_data)
  return _tensor(out, self.dtype)

def rmul_tensor_ops(self, other):
  return mul_tensor_ops(other, self)

def div_tensor_ops(self, other):
  from .._core import _tensor
  other = _ensure_tensor(other, self)
  target_shape, _ = broadcast_shape(self.shape, other.shape)
  self_data, other_data = broadcast(self.data, target_shape), broadcast(other.data, target_shape)
  out = div_tensor(self_data, other_data)
  return _tensor(out, self.dtype)

def rdiv_tensor_ops(self, other):
  return div_tensor_ops(other, self)

def register_binary_operators():
  from .._core import _tensor
  _tensor.__add__ = add_tensor_ops
  _tensor.__radd__ = radd_tensor_ops
  _tensor.__sub__ = sub_tensor_ops
  _tensor.__rsub__ = rsub_tensor_ops
  _tensor.__mul__ = mul_tensor_ops
  _tensor.__rmul__ = rmul_tensor_ops
  _tensor.__truediv__ = div_tensor_ops
  _tensor.__rtruediv__ = rdiv_tensor_ops