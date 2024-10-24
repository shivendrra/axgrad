from .._tensor import tensor
from typing import List
from ..helpers.ops import dedup

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