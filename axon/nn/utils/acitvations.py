from ...helpers.acitvations import *
from ...tensor import tensor
from ...axgrad import backward
from ..module import Module

class ReLU(Module):
  def __init__(self, inplace=False):
    super().__init__()
    self.inplace = inplace

  def forward(self, x:tensor):
    def _apply(x):
      if isinstance(x, list):
        return [_apply(d) for d in x]
      else:
        return relu(x)
    if self.inplace:
      x.data = _apply(x.data)
      x._ops = '<relu>'
      x._backward = backward.relu_back(x, x)
      return x
    else:
      out = tensor(_apply(x.data), child=(self,), _ops='<relu>')
      out._backward = backward.relu_back(x, out)
      return out

  def __call__(self, x:tensor):
    return self.forward(x)
  
  def __repr__(self):
    return f"<Relu(inplace={self.inplace})>"

class Tanh(Module):
  def __init__(self, inplace=False):
    super().__init__()
    self.inplace = inplace

  def forward(self, x:tensor):
    def _apply(x):
      if isinstance(x, list):
        return [_apply(d) for d in x]
      else:
        return tanh(x)
    if self.inplace:
      x.data = _apply(x.data)
      x._ops = '<tanh>'
      x._backward = backward.tanh_back(x, x)
      return x
    else:
      out = tensor(_apply(x.data), child=(self,), _ops='<tanh>')
      out._backward = backward.tanh_back(x, out)
      return out

  def __call__(self, x:tensor):
    return self.forward(x)
  
  def __repr__(self):
    return f"<Tanh(inplace={self.inplace})>"

class Sigmoid(Module):
  def __init__(self, inplace=False):
    super().__init__()
    self.inplace = inplace

  def forward(self, x:tensor):
    def _apply(x):
      if isinstance(x, list):
        return [_apply(d) for d in x]
      else:
        return sigmoid(x)
    if self.inplace:
      x.data = _apply(x.data)
      x._ops = '<sigmoid>'
      x._backward = backward.sigmoid_back(x, x)
      return x
    else:
      out = tensor(_apply(x.data), child=(self,), _ops='<sigmoid>')
      out._backward = backward.sigmoid_back(x, out)
      return out

  def __call__(self, x:tensor):
    return self.forward(x)
  
  def __repr__(self):
    return f"<Sigmoid(inplace={self.inplace})>"

class GELU(Module):
  def __init__(self, inplace=False):
    super().__init__()
    self.inplace = inplace

  def forward(self, x:tensor):
    def _apply(x):
      if isinstance(x, list):
        return [_apply(d) for d in x]
      else:
        return gelu(x)
    if self.inplace:
      x.data = _apply(x.data)
      x._ops = '<gelu>'
      x._backward = backward.gelu_back(x, x)
      return x
    else:
      out = tensor(_apply(x.data), child=(self,), _ops='<gelu>')
      out._backward = backward.gelu_back(x, out)
      return out

  def __call__(self, x:tensor):
    return self.forward(x)
  
  def __repr__(self):
    return f"<GELU(inplace={self.inplace})>"

class LeakyRelu(Module):
  def __init__(self, inplace=False):
    super().__init__()
    self.inplace = inplace

  def forward(self, x:tensor):
    def _apply(x):
      if isinstance(x, list):
        return [_apply(d) for d in x]
      else:
        return LeakyRELU(x)
    if self.inplace:
      x.data = _apply(x.data)
      x._ops = '<LeakyRelu>'
      x._backward = backward.leaky_r_backward(x, x)
      return x
    else:
      out = tensor(_apply(x.data), child=(self,), _ops='<LeakyRelu>')
      out._backward = backward.leaky_r_backward(x, out)
      return out

  def __call__(self, x:tensor):
    return self.forward(x)
  
  def __repr__(self):
    return f"<LeakyRelu(inplace={self.inplace})>"