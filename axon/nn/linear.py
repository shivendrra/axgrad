from ..ops import matmul
from ..tensor import tensor
from ..helpers.statics import randn
from .module import Module

class Linear(Module):
  def __init__(self, _in, _out, bias=False):
    self.w = tensor(randn((_in, _out)))
    self.b = tensor([0 for _ in range(_out)]) if bias is True else None

  def __call__(self, x):
    return self.forward(x)

  def forward(self, x):
    x = x if isinstance(x, tensor) else tensor(x)
    if len(x.data[0]) == len(self.w.data[0]):
      out = matmul(x, self.w.transpose())
      if self.b is not None:
        out = [[out.data[i][j] + self.b.data[j] for j in range(len(out.data[0]))] for i in range(len(out.data))]
      return tensor(out)
    else:
      raise ValueError(f"Tensor shape error: {len(x.data[0])} != {len(self.w.data[0])}")

  def parameters(self):
    params = self.w.flatten()
    if self.b is not None:
      params.extend(self.b.flatten())
    return params

  def __repr__(self) -> str:
    return f"linear layer"