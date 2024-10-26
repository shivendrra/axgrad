from .._tensor import tensor
from typing import List
from ..helpers.ops import dedup, compute_norm
from ..helpers.utils import _zeros_like
from .._grad import grads

class Optimizer:
  """ Base class for all optimizers. """
  def __init__(self, params:List[tensor], lr:float) -> None:
    for x in params:
      if x.requires_grad is None: x.requires_grad = True
    
    self.params: List[tensor] = dedup([x for x in params if x.requires_grad])
    self.buffers: List[tensor] = dedup([x for x in params if not x.requires_grad])
    self.lr = lr

  def zero_grad(self):
    for params in self.params:
      params.zero_grad()

  def step(self):
    assert self.params[0].training_mode == True, (
            f"""tensor.training={tensor.training}, tensor.training must be enabled to use the optimizer.
                - help: Consider setting tensor.training=True before calling Optimizer.step().""")
    for param in self.params:
      if param.grad is None:
        continue
      self._step(param)
  def _step(self, param) -> tensor: raise NotImplementedError

class SGD(Optimizer):
  """
  basic Stochastic Gradient Descent (SGD) optimizer.
  `classic` is a boolean flag that determines whether to use the popular momentum update rule or the classic momentum update rule.
  """
  def __init__(self, parameters:tensor, lr:float=0.01):
    super().__init__(parameters, lr)
    self.parameters, self.lr = parameters, lr
  
  def _step(self, param):
    param.data = (param.data - param.grad._scalar_mul(self.lr)).data
    return param

class LARS(Optimizer):
  def __init__(self, parameters, lr, momentum: float = 0, eta: float = 1e-3,
               dampening: float = 0, weight_decay: float = 0, nesterov: bool = False, eps: float = 0):
    super().__init__(parameters, lr)
    if momentum < 0.0:
      raise ValueError("Invalid momentum value: {}".format(momentum))
    if weight_decay < 0.0:
      raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
    
    self.defaults = dict(lr=lr, momentum=momentum, eta=eta, dampening=dampening,
                         weight_decay=weight_decay, nesterov=nesterov, eps=eps)
    
    if nesterov and (momentum <= 0 or dampening != 0):
      raise ValueError("Nesterov momentum requires a momentum and zero dampening")
    
    # initialize momentum buffers
    self.momentum_buffers = {param: tensor(_zeros_like(param.data), requires_grad=False) for param in self.params}

  def _step(self, param):
    """ Performs a single optimization step. """
    if param.grad is None:
      return
    
    # compute the L2 norm of the parameter and the gradient
    weight_norm = compute_norm(param.data, p=2)  # L2 norm of the weights
    grad_norm = compute_norm(param.grad.data, p=2)  # L2 norm of the gradient

    # compute the effective learning rate
    if weight_norm > 0 and grad_norm > 0:
      lr = self.defaults['lr'] * (weight_norm / (grad_norm + self.defaults['eps']))
    else:
      lr = self.defaults['lr']  # fallback to the default learning rate if norms are zero

    # apply weight decay if specified
    if self.defaults['weight_decay'] > 0:
      param.grad += self.defaults['weight_decay'] * param.data

    # update momentum buffer
    momentum_buffer = self.momentum_buffers[param]
    momentum_buffer = (self.defaults['momentum'] * momentum_buffer) + param.grad
    self.momentum_buffers[param] = momentum_buffer

    # Nesterov momentum
    if self.defaults['nesterov']:
      param.data -= lr * (momentum_buffer + param.grad)
    else:
      param.data -= lr * momentum_buffer

    return param