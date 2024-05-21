from ..ops import matmul
from ..tensor import tensor
from ..helpers.statics import ones, zeros

class Linear:
  def __init__(self, _in:int, _out:int, bias:bool=True) -> tensor:
    self.wei = 0
    self.b = zeros((1, _out), dtype=float) if bias is True else None

  def forward(self, x:tensor):
    x = x if isinstance(x, tensor) else tensor(x)
    out = matmul(x, self.wei.transpose()) + self.b
    return out
  
  def __call__(self, x:tensor) -> tensor:
    return self.forward(x)
  
  def __repr__(self) -> str:
    return f"linear layer"