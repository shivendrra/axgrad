from ...helpers.utils import _zeros_like
from ...helpers.ops import sum_axis0
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

class __LAYERNORM__:
  def __init__(self, gamma, beta, bias, out, elem_aff, eps, x, mean, var): self.gamma, self.beta, self.bias, self.out, self.elem_aff, self.eps, self.x, self.mean, self.var = gamma, beta, bias, out, elem_aff, eps, x, mean, var
  def backward(self, grad):
    N = self.x.shape[-1]

    # grads w.r.t. the normalized input
    x_hat = (self.x - self.mean) / (self.var + [self.eps]).sqrt()
    dx_hat = grad * self.gamma if self.elem_aff else grad

    # grads w.r.t. the input (x)
    dvar = ((dx_hat * (self.x - self.mean)) * [-0.5] * (self.var + [self.eps])**-1.5).sum(axis=-1, keepdims=True)
    dmean = (dx_hat * [-1] / (self.var + [self.eps]).sqrt()).sum(axis=-1, keepdims=True) \
            + dvar * [-2] * (self.x - self.mean).mean(axis=-1, keepdims=True)

    dx = dx_hat / (self.var + [self.eps]).sqrt()
    dx += ((self.x - self.mean) * dvar.unsqueeze() * [2]) / [N]
    dx += dmean.unsqueeze() / [N]
    return dx, x_hat

  def __call__(self) -> Callable:
    dx, x_hat = self.backward(self.out.grad.data)
    self.x.grad.data = dx.data
    
    if self.elem_aff:
      dgamma = sum_axis0((self.out.grad * x_hat).data)
      dbeta = sum_axis0(self.out.grad.data)
      self.gamma.grad.data = dgamma
      self.beta.grad.data = dbeta

    if self.bias is not None:
      self.bias.grad.data = sum_axis0(self.out.grad.data)

    return self.__call__

class __BATCHNORM__:
  def __init__(self, gamma, beta, running_mean, running_var, out, affine, eps, x, mean, var): self.gamma, self.beta, self.running_mean, self.running_var, self.out, self.affine, self.eps, self.x, self.mean, self.var = gamma, beta, running_mean, running_var, out, affine, eps, x, mean, var
  def backward(self, grad):
    N, _ = self.x.shape

    x_hat = (self.x - self.mean) / (self.var + [self.eps]).sqrt()
    dx_hat = grad * self.gamma if self.affine else grad

    dvar = (dx_hat * (self.x - self.mean) * [-0.5] * (self.var + [self.eps]) ** -1.5).sum(axis=0, keepdims=True)
    dmean = (dx_hat * [-1] / (self.var + [self.eps]).sqrt()).sum(axis=0, keepdims=True) \
            + dvar * [-2] * (self.x - self.mean).mean(axis=0, keepdims=True)
    dx = dx_hat / (self.var + [self.eps]).sqrt()
    dx += ((self.x - self.mean) * dvar.unsqueeze().T * [2]) / [N]
    dx += dmean.unsqueeze().T / [N]
    return dx, x_hat

  def __call__(self):
    dx, x_hat = self.backward(self.out.grad.data)
    self.x.grad.data = dx.data
    
    if self.affine:
      dgamma = sum_axis0((self.out.grad * x_hat).data)
      dbeta = sum_axis0(self.out.grad.data)
      self.gamma.grad.data = dgamma
      self.beta.grad.data = dbeta

    return self.__call__

class __RMSNORM__:
  def __init__(self, wei, out, eps, x):
    self.wei, self.out, self.eps, self.x = wei, out, eps, x

  def backward(self, grad):
    mean_square = (self.x ** 2).mean(axis=-1, keepdims=True)
    rms = (mean_square + [self.eps]).sqrt()

    # grads w.r.t. `x`
    grad_x = grad * self.wei / rms.unsqueeze()
    grad_rms = (grad * -self.wei * self.x / (rms ** 3).unsqueeze()).mean(axis=-1, keepdims=True)
    dx = grad_x + (grad_rms.unsqueeze() * self.x)

    # grads w.r.t. `wei`
    dwei = (grad * self.x / rms.unsqueeze()).sum(axis=0, keepdims=True)
    return dx, dwei

  def __call__(self):
    dx, dwei = self.backward(self.out.grad.data)
    self.x.grad.data = dx.data
    self.wei.grad.data = dwei.data
    return self.__call__
