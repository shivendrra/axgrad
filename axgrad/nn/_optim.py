from .._tensor import tensor
from typing import List
from ..helpers.ops import dedup

# class Optimizer:
#   def __init__(self, params:List[tensor], lr:float) -> None:
#     for x in params:
#       if x.requires_grad is None: x.requires_grad = True
    
#     self.params: List[tensor] = dedup([x for x in params if x.requires_grad])
#     self.buffers: List[tensor] = dedup([x for x in params if not x.requires_grad])
#     self.lr = lr
  
#   def step(self):
#     """
#     Performs a single optimization step.
#     """
#     tensor.realize(*self.schedule_step())
#   def schedule_step(self) -> List[tensor]:
#     """
#     Returns the tensors that need to be realized to perform a single optimization step.
#     """
#     assert tensor.training, (
#             f"""Tensor.training={tensor.training}, Tensor.training must be enabled to use the optimizer.
#                 - help: Consider setting Tensor.training=True before calling Optimizer.step().""")
#     return self._step()+self.params+self.buffers
#   def _step(self) -> List[tensor]: raise NotImplementedError

class Optim:
  """
  basic Stochastic Gradient Descent (SGD) optimizer.
  `classic` is a boolean flag that determines whether to use the popular momentum update rule or the classic momentum update rule.
  """
  class SGD:
    def __init__(self, parameters, lr=0.01): self.parameters, self.lr = parameters, lr
    def step(self):
      def _mul(grad): return [_mul(g) for g in grad] if isinstance(grad, list) else grad * self.lr
      def _sub(param, grad): return [_sub(p, g) for p, g in zip(param, grad)] if isinstance(grad, list) else param - grad

      for param in self.parameters:
        if param.grad is not None:
          param.data = _sub(param.data, _mul(param.grad.data))