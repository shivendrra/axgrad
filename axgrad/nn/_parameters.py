"""
  @nn/_paramters.py generates parameter for nn blocks
  @brief inputs a tensor and converts it into a axgrad.tensor with ``requires_grad = true``
  @comments:
  - none...
"""

from .._tensor import tensor
from ..helpers.shape import flatten

class Parameter(tensor):
  def __init__(self, data) -> None:
    data = data
    super().__init__(data, dtype='float32', requires_grad=True)
  
  def zero_grad(self) -> None:
    self.grad = None
  
  def tolist(self) -> list:
    return super().tolist()
  
  def numel(self) -> int:
    return len(flatten(self.data))
  
  def __repr__(self) -> str:
    return "\nParameter containing:\n" + super().__repr__()
  
  def __str__(self) -> str:
    return "\nParameters :\n" + super().__str__()