from ..tensor import tensor
from ..helpers.utils import generate_random_list
from ..helpers.shape import _flatten

class Parameter(tensor):
  def __init__(self, shape):
    data = generate_random_list(shape)
    super().__init__(data)
  
  def zero_grad(self):
    self.grad = 0
  
  def tolist(self):
    return super().tolist()
  
  def numel(self):
    return len(_flatten(self.data))

  def __repr__(self):
    return "\nParameter containig:\n" + super().__repr__()