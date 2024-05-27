from ...helpers.acitvations import *
from ...tensor import tensor
from ...axgrad import backward

class ReLU:
  def __call__(self, x:tensor):
    def _apply(x):
      if isinstance(x, list):
        return [_apply(d) for d in x]
      else:
        return relu(x)
    out = tensor(_apply(x.data), child=(self,), _ops='<relu>')
    out._backward = backward.relu_back(x, out)
    return out

class Tanh:
  def __call__(self, x:tensor):
    def _apply(x):
      if isinstance(x, list):
        return [_apply(d) for d in x]
      else:
        return tanh(x)
    out = tensor(_apply(x.data), child=(self,), _ops='<tanh>')
    out._backward = backward.tanh_back(x, out)
    return out

class Sigmoid:
  def __call__(self, x:tensor):
    def _apply(x):
      if isinstance(x, list):
        return [_apply(d) for d in x]
      else:
        return sigmoid(x)
    out = tensor(_apply(x.data), child=(self,), _ops='<sigmoid>')
    out._backward = backward.sigmoid_back(x, out)
    return out

class GELU:
  def __call__(self, x:tensor):
    def _apply(x):
      if isinstance(x, list):
        return [_apply(d) for d in x]
      else:
        return gelu(x)
    out = tensor(_apply(x.data), child=(self,), _ops='<gelu>')
    out._backward = backward.gelu_back(x, out)
    return out

class LeakyRelu:
  def __call__(self, x:tensor):
    def _apply(x):
      if isinstance(x, list):
        return [_apply(d) for d in x]
      else:
        return LeakyRELU(x)
    out = tensor(_apply(x.data), child=(self,), _ops='<leakyrelu>')
    out._backward = backward.leaky_r_backward(x, out)
    return out