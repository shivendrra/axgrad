from typing import Callable
from ...helpers.ops import matmul, _apply_padding, _conv2d
from ...helpers.shape import transpose, get_shape
from ...helpers.utils import _zeros_like

def sum_to_shape(grad, shape):
  for i, (grad_dim, shape_dim) in enumerate(zip(get_shape(grad), shape)):
    if grad_dim > shape_dim:
      grad = sum(grad, axis=i)
  return grad

class __ADD__:
  def __init__(self, first, second, out) -> None: self.first, self.second, self.out = first, second, out
  def backward(self, grad, out):
    if not isinstance(grad, list):
      grad += out
      return grad
    return [self.backward(g, og) for g, og, in zip(grad, out)]
  
  def __call__(self) -> Callable:
    self.first.grad.data = self.backward(self.first.grad.data, self.out.grad.data)
    self.second.grad.data = self.backward(self.second.grad.data, self.out.grad.data)
    return self.__call__

class __MUL__:
  def __init__(self, first, second, out) -> None: self.first, self.second, self.out = first, second, out
  def backward(self, grad, out, mul):
    if not isinstance(grad, list):
      grad += out * mul
      return grad
    return [self.backward(g, og, m) for g, og, m in zip(grad, out, mul)]

  def __call__(self) -> Callable:
    self.first.grad.data = self.backward(self.first.grad.data, self.out.grad.data, self.second.data)
    self.second.grad.data = self.backward(self.second.grad.data, self.out.grad.data, self.first.data)
    return self.__call__

class __MATMUL__:
  def __init__(self, first, second, out) -> None: self.first, self.second, self.out = first, second, out
  def backward(self, grad, out):
    if not isinstance(grad, list):
      grad += out
      return grad
    return [self.backward(g, og) for g, og in zip(grad, out)]

  def __call__(self) -> Callable:
    grad_A = matmul(self.out.grad.data, transpose(self.second.data))
    grad_B = matmul(transpose(self.first.data), self.out.grad.data)

    if get_shape(self.first.data) != get_shape(grad_A):
      grad_A = sum_to_shape(grad_A, get_shape(self.first.data))
    if get_shape(self.second.data) != get_shape(grad_B):
      grad_B = sum_to_shape(grad_B, get_shape(self.second.data))

    self.first.grad.data = self.backward(self.first.grad.data, grad_A)
    self.second.grad.data = self.backward(self.second.grad.data, grad_B)
    return self.__call__

class __CONV2D__:
  def __init__(self, input_tensor, kernel, out, stride=1, padding=0): self.input_tensor, self.kernel, self.out, self.stride, self.padding = input_tensor, kernel, out, stride, padding
  def backward(self, grad, out):
    if not isinstance(grad, list):
      grad += out
      return grad
    return [self.backward(g, og) for g, og in zip(grad, out)]

  def __call__(self) -> Callable:
    padded_input = _apply_padding(self.input_tensor.data, self.padding)

    grad_input = self._conv2d_backprop_input(self.out.grad.data, self.kernel.data)
    grad_kernel = self._conv2d_backprop_kernel(padded_input, self.out.grad.data)

    if get_shape(grad_input) != get_shape(self.input_tensor.data):
      grad_input = sum_to_shape(grad_input, get_shape(self.input_tensor.data))
    if get_shape(grad_kernel) != get_shape(self.kernel.data):
      grad_kernel = sum_to_shape(grad_kernel, get_shape(self.kernel.data))

    self.input_tensor.grad.data = self.backward(self.input_tensor.grad.data, grad_input)
    self.kernel.grad.data = self.backward(self.kernel.grad.data, grad_kernel)

    return self.__call__

  def _conv2d_backprop_input(self, grad_output, kernel):
    flipped_kernel = [row[::-1] for row in kernel[::-1]]
    grad_input = _conv2d(grad_output, flipped_kernel, stride=1)
    return grad_input

  def _conv2d_backprop_kernel(self, padded_input, grad_output):
    grad_kernel = _zeros_like(self.kernel.data)
    kernel_h, kernel_w = len(grad_kernel), len(grad_kernel[0])

    for i in range(kernel_h):
      for j in range(kernel_w):
        for m in range(len(grad_output)):
          for n in range(len(grad_output[0])):
            padded_i = i + m * self.stride
            padded_j = j + n * self.stride
            if padded_i < len(padded_input) and padded_j < len(padded_input[0]):
              grad_kernel[i][j] += (
                padded_input[padded_i][padded_j] * grad_output[m][n]
              )
    return grad_kernel

class __POW__:
  def __init__(self, first, out, power) -> None: self.first, self.out, self.power = first, out, power
  def backward(self, grad, out, power):
    if not isinstance(grad, list):
      grad += (power * out ** (power - 1)) * out
      return grad
    return [self.backward(g, og, power) for g, og in zip(grad, out)]
  
  def __call__(self) -> Callable:
    self.first.grad.data = self.backward(self.first.grad.data, self.out.grad.data, self.power)
    return self.__call__

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