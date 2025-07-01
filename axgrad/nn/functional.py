from .loss import MSELoss, MAELoss
from ..tensor import Tensor
from .module import Module

def mse(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor: return MSELoss(reduction)(pred, target)
def mae(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor: return MAELoss(reduction)(pred, target)

class Tanh(Module):
  def __init__(self, inplace: Tensor = False):
    self.inplace = inplace
    super().__init__()
  def forward(self, x: Tensor) -> Tensor: return x.tanh()
  def __repr__(self) -> str: return f"Tanh(inplace={self.inplace})"
  def inner_repr(self) -> str: return f"inplace={self.inplace}"