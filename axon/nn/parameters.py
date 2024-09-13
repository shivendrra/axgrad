from ..tensor import tensor
from ..helpers.utils import _randn
from ..helpers.shape import flatten

class Parameter(tensor):
  def __init__(self, data) -> None:
    data = data
    super().__init__(data, dtype='float32', requires_grad=True)
  
  def zero_grad(self) -> None:
    self.grad = 0
  
  def tolist(self) -> list:
    return super().tolist()
  
  def numel(self) -> int:
    return len(flatten(self.data))
  
  def __repr__(self) -> str:
    return "\nParameter containing:\n" + super().__repr__()
  
  def __str__(self) -> str:
    return "\nParameters :\n" + super().__str__()