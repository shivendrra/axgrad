from .._tensor import tensor
from ._parameters import Parameter
from ._module import Module
from ..helpers.utils import _randn, _zeros

class Linear(Module):
  def __init__(self, _in, _out, bias=False):
    super(Linear, self).__init__()
    self.wei = Parameter(_randn(shape=(_in, _out)))
    if bias:
      self.bias = Parameter(_zeros(shape=(1, _out)))
    else:
      self.bias = None

  def __call__(self, x):
    return self.forward(x)

  def forward(self, x):
    x = x if isinstance(x, tensor) else tensor(x, dtype=tensor.float32, requires_grad=True)
    out = x @ self.wei
    if self.bias is not None:
      out = out + self.bias
    return out

  def parameters(self):
    params = [self.wei]
    if self.bias is not None:
      params.append(self.bias)
    return params
  
  def __repr__(self):
    return f"<LinearLayer in_features={self.wei.shape[0]} out_features={self.wei.shape[1]}>"