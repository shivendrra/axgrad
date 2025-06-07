from .main import add_tensor, sub_tensor, mul_tensor, div_tensor
from .._dtype import Dtype
from ..helpers.shape import broadcast, broadcast_shape, get_shape
from .._core import _tensor

def add_tensor_ops(tensor1, tensor2):
  target_shape, requires_broadcasting = broadcast_shape(tensor1.shape, tensor2.shape)
  if requires_broadcasting:
    tensor1.data, tensor1.grad, tensor1.shape = Dtype.handle_conversion(broadcast(tensor1.data, target_shape), tensor1.dtype), broadcast(tensor1.grad, target_shape), get_shape(tensor1.data)
    tensor2.data, tensor2.grad, tensor2.shape = Dtype.handle_conversion(broadcast(tensor2.data, target_shape), tensor2.dtype), broadcast(tensor2.grad, target_shape), get_shape(tensor2.data)
  if tensor1.size == tensor2.size:
    out = add_tensor(tensor1.data, tensor2.data)
  return _tensor(out, tensor1.dtype)

def sub_tensor_ops(tensor1, tensor2):
  target_shape, requires_broadcasting = broadcast_shape(tensor1.shape, tensor2.shape)
  if requires_broadcasting:
    tensor1.data, tensor1.grad, tensor1.shape = Dtype.handle_conversion(broadcast(tensor1.data, target_shape), tensor1.dtype), broadcast(tensor1.grad, target_shape), get_shape(tensor1.data)
    tensor2.data, tensor2.grad, tensor2.shape = Dtype.handle_conversion(broadcast(tensor2.data, target_shape), tensor2.dtype), broadcast(tensor2.grad, target_shape), get_shape(tensor2.data)
  if tensor1.size == tensor2.size:
    out = sub_tensor(tensor1.data, tensor2.data)
  return _tensor(out, tensor1.dtype)

def mul_tensor_ops(tensor1, tensor2):
  target_shape, requires_broadcasting = broadcast_shape(tensor1.shape, tensor2.shape)
  if requires_broadcasting:
    tensor1.data, tensor1.grad, tensor1.shape = Dtype.handle_conversion(broadcast(tensor1.data, target_shape), tensor1.dtype), broadcast(tensor1.grad, target_shape), get_shape(tensor1.data)
    tensor2.data, tensor2.grad, tensor2.shape = Dtype.handle_conversion(broadcast(tensor2.data, target_shape), tensor2.dtype), broadcast(tensor2.grad, target_shape), get_shape(tensor2.data)
  if tensor1.size == tensor2.size:
    out = mul_tensor(tensor1.data, tensor2.data)
  return _tensor(out, tensor1.dtype)

def div_tensor_ops(tensor1, tensor2):
  target_shape, requires_broadcasting = broadcast_shape(tensor1.shape, tensor2.shape)
  if requires_broadcasting:
    tensor1.data, tensor1.grad, tensor1.shape = Dtype.handle_conversion(broadcast(tensor1.data, target_shape), tensor1.dtype), broadcast(tensor1.grad, target_shape), get_shape(tensor1.data)
    tensor2.data, tensor2.grad, tensor2.shape = Dtype.handle_conversion(broadcast(tensor2.data, target_shape), tensor2.dtype), broadcast(tensor2.grad, target_shape), get_shape(tensor2.data)
  if tensor1.size == tensor2.size:
    out = div_tensor(tensor1.data, tensor2.data)
  return _tensor(out, tensor1.dtype)

def register_binary_operators():
  from .._core import _tensor

  _tensor.__add__ = add_tensor_ops
  _tensor.__sub__ = sub_tensor_ops
  _tensor.__mul__ = mul_tensor_ops
  _tensor.__div__ = div_tensor_ops