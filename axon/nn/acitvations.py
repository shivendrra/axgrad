from ..helpers.acitvations import *
from ..tensor import tensor

class ReLU:
  def __call__(self, x:tensor):
    def _apply(x):
      if isinstance(x, list):
        return [_apply(d) for d in x]
      else:
        return relu(x)
    out = _apply(x.data)
    return tensor(out, child=(self,), _ops='<ReLU>')

class Tanh:
  def __call__(self, x:tensor):
    def _apply(x):
      if isinstance(x, list):
        return [_apply(d) for d in x]
      else:
        return tanh(x)
    out = _apply(x.data)
    return tensor(out, child=(self,), _ops='<tanh>')

class Sigmoid:
  def __call__(self, x:tensor):
    def _apply(x):
      if isinstance(x, list):
        return [_apply(d) for d in x]
      else:
        return sigmoid(x)
    out = _apply(x.data)
    return tensor(out, child=(self,), _ops='<sigmoid>')

class GELU:
  def __call__(self, x:tensor):
    def _apply(x):
      if isinstance(x, list):
        return [_apply(d) for d in x]
      else:
        return gelu(x)
    out = _apply(x.data)
    return tensor(out, child=(self,), _ops='<gelu>')