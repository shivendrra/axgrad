from ...helpers.functionals import *
from ...autograd._backward import *
from ...tensor import tensor
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
      x.prev = (self, )
      x.grad_fn = "<ReluBackward>"
      x._backward = backward.relu_back(x, x)
      return x
    else:
      out = tensor(_apply(x.data), requires_grad=True, dtype='float32')
      out.prev = (self, )
      out.grad_fn = "<ReluBackward>"
      out._backward = backward.relu_back(x, out)
      return out

  def __call__(self, x:tensor):
    x = x if isinstance(x, tensor) else tensor(x, requires_grad=True, dtype='float32')
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
      x.prev = (self, )
      x.grad_fn = "<TanhBackward>"
      x._backward = backward.tanh_back(x, x)
      return x
    else:
      out = tensor(_apply(x.data), requires_grad=True, dtype='float32')
      out.prev = (self, )
      out.grad_fn = "<TanhBackward>"
      out._backward = backward.tanh_back(x, out)
      return out

  def __call__(self, x:tensor):
    x = x if isinstance(x, tensor) else tensor(x, requires_grad=True, dtype='float32')
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
      x.prev = (self, )
      x.grad_fn = "<SigmoidBackward>"
      x._backward = backward.sigmoid_back(x, x)
      return x
    else:
      out = tensor(_apply(x.data), requires_grad=True, dtype='float32')
      out.prev = (self, )
      out.grad_fn = "<SigmoidBackward>"
      out._backward = backward.sigmoid_back(x, out)
      return out

  def __call__(self, x:tensor):
    x = x if isinstance(x, tensor) else tensor(x, requires_grad=True, dtype='float32')
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
      x.prev = (self, )
      x.grad_fn = "<GeluBackward>"
      x._backward = backward.gelu_back(x, x)
      return x
    else:
      out = tensor(_apply(x.data), requires_grad=True, dtype='float32')
      out.prev = (self, )
      out.grad_fn = "<GeluBackward>"
      out._backward = backward.gelu_back(x, out)
      return out

  def __call__(self, x:tensor):
    x = x if isinstance(x, tensor) else tensor(x, requires_grad=True, dtype='float32')
    return self.forward(x)
  
  def __repr__(self):
    return f"<GELU(inplace={self.inplace})>"

class LeakyRELU(Module):
  def __init__(self, inplace=False):
    super().__init__()
    self.inplace = inplace

  def forward(self, x:tensor):
    def _apply(x):
      if isinstance(x, list):
        return [_apply(d) for d in x]
      else:
        return LeakyRelu(x)
    if self.inplace:
      x.data = _apply(x.data)
      x.prev = (self, )
      x.grad_fn = "<LeakyReluBackward>"
      x._backward = backward.leaky_r_back(x, x)
      return x
    else:
      out = tensor(_apply(x.data), requires_grad=True, dtype='float32')
      out.prev = (self, )
      out.grad_fn = "<LeakyReluBackward>"
      out._backward = backward.leaky_r_back(x, out)
      return out

  def __call__(self, x:tensor):
    x = x if isinstance(x, tensor) else tensor(x, requires_grad=True, dtype='float32')
    return self.forward(x)
  
  def __repr__(self):
    return f"<LeakyRelu(inplace={self.inplace})>"

class SiLU(Module):
  def __init__(self, inplace=False):
    super().__init__()
    self.inplace = inplace

  def forward(self, x:tensor):
    def _apply(x):
      if isinstance(x, list):
        return [_apply(d) for d in x]
      else:
        return silu(x)
    if self.inplace:
      x.data = _apply(x.data)
      x.prev = (self, )
      x.grad_fn = "<SiluBackward>"
      x._backward = backward.silu_back(x, x)
      return x
    else:
      out = tensor(_apply(x.data), requires_grad=True, dtype='float32')
      out.prev = (self, )
      out.grad_fn = "<SiluBackward>"
      out._backward = backward.silu_back(x, out)
      return out

  def __call__(self, x:tensor):
    x = x if isinstance(x, tensor) else tensor(x, requires_grad=True, dtype='float32')
    return self.forward(x)
  
  def __repr__(self):
    return f"<LeakyRelu(inplace={self.inplace})>"