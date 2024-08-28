from ..tensor import tensor
from ..helpers.utils import _randn
from ..helpers.shape import flatten

class Parameter(tensor):
  def __init__(self, shape) -> None:
    data = _randn(domain=(-1, 1), shape=shape)
    super().__init__(data)
  
  def zero_grad(self) -> None:
    self.grad = 0
  
  def tolist(self) -> list:
    return super().tolist()
  
  def numel(self) -> int:
    return len(flatten(self.data))
  
  def __repr__(self) -> str:
    return "\nParameter containing:\n" + super().__repr__()