from ...helpers.utils import _zeros_like
from typing import Callable

class __STACK__:
  def __init__(self, out, tensors, axis): self.out, self.tensors, self.axis = out, tensors, axis
  def __call__(self) -> Callable:
    split_grads = self._split_grad(self.out.grad.data)
    for tensor, grad_part in zip(self.tensors, split_grads):
      if tensor.grad is None:
        tensor.grad = grad_part
      else:
        tensor.grad += grad_part

  def _split_grad(self, grad):
    return [
      grad[i] if self.axis == 0 else grad[(slice(None),) * self.axis + (i,)]
      for i in range(len(self.tensors))
    ]

class __CONCAT__:
  def __init__(self, out, tensors, axis): self.out, self.tensors, self.axis = out, tensors, axis
  def __call__(self):
    split_grads = self._split_grad(self.out.grad.data)
    for tensor, grad_part in zip(self.tensors, split_grads):
      if tensor.grad is None:
        tensor.grad = grad_part
      else:
        tensor.grad += grad_part

  def _split_grad(self, grad):
    split_grads, current_index = [], 0
    for tensor in self.tensors:
      tensor_size = tensor.shape[self.axis]
      split_grads.append(grad[current_index:current_index + tensor_size])
      current_index += tensor_size
    return split_grads

class __EMBEDD__:
  def __init__(self, first, indices, out): self.first, self.indices, self.out = first, indices, out
  def backward(self, grad, idx, output):
    def _apply_grad(grad, out):
      if isinstance(grad, list):
        return [_apply_grad(g, o) for g, o in zip(grad, out)]
      grad += out
      return grad
    grad_weight = _zeros_like(grad)
    for i, idx in enumerate(idx):
      grad_weight[idx] = _apply_grad(grad_weight[idx], output[i])
    return grad_weight
  def __call__(self) -> Callable:
    self.first.grad.data = self.backward(self.first.data, self.indices, self.out.data)
    return self.__call__