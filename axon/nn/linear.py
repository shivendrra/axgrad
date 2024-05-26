from ..ops import matmul
from ..tensor import tensor
from .parameter import Parameter
from .module import Module

class Linear(Module):
  def __init__(self, _in, _out, bias=False):
    super(Linear, self).__init__()
    self.w = Parameter(shape=(_in, _out))
    if bias:
      self.b = Parameter(shape=(1, _out))
    else:
      self.b = None

  def __call__(self, x):
    return self.forward(x)

  def forward(self, x):
    x = x if isinstance(x, tensor) else tensor(x)
    out = matmul(x, self.w)
    if self.b is not None:
      out.data = [[out.data[i][j] + self.b.data[0][j] for j in range(len(out.data[0]))] for i in range(len(out.data))]
    return tensor(out, child=(x, self.w, self.b), _ops='Linear')

  def parameters(self):
    params = [self.w]
    if self.b is not None:
        params.append(self.b)
    return params
  
  def __repr__(self):
    return f"<LinearLayer in_features={self.w.shape[0]} out_features={self.w.shape[1]}>"