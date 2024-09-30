from ..._tensor import tensor
from ...autograd._backward import Backward
from .._module import Module
from ...helpers.functionals import *

class ReLU(Module):
  def __init__(self, inplace=False):
    super().__init__()
    self.inplace = inplace

  def forward(self, x:tensor):
    def _apply(data):
      return [_apply(d) for d in data] if isinstance(data, list) else relu(data)
    if self.inplace:
      x.data, x.prev, x.grad_fn, x._backward = _apply(x.data), (self, ), "<ReluBackwards>", Backward.relu_backwards(x, x)
      return x
    else:
      out = tensor(_apply(x.data), x.requires_grad, x.dtype)
      out.prev, out.grad_fn, out._backward = (x, ), "<ReluBackwards>", Backward.relu_backwards(x, out)
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
    def _apply(data):
      return [_apply(d) for d in data] if isinstance(data, list) else tanh(data)
    if self.inplace:
      x.data, x.prev, x.grad_fn, x._backward = _apply(x.data), (self, ), "<TanhBackwards>", Backward.tanh_backwards(x, x)
      return x
    else:
      out = tensor(_apply(x.data), x.requires_grad, x.dtype)
      out.prev, out.grad_fn, out._backward = (x, ), "<TanhBackwards>", Backward.tanh_backwards(x, out)
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
    def _apply(data):
      return [_apply(d) for d in data] if isinstance(data, list) else sigmoid(data)
    if self.inplace:
      x.data, x.prev, x.grad_fn, x._backward = _apply(x.data), (self, ), "<SigmoidBackwards>", Backward.sigmoid_backwards(x, x)
      return x
    else:
      out = tensor(_apply(x.data), x.requires_grad, x.dtype)
      out.prev, out.grad_fn, out._backward = (x, ), "<SigmoidBackwards>", Backward.sigmoid_backwards(x, out)
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
    def _apply(data):
      return [_apply(d) for d in data] if isinstance(data, list) else gelu(data)
    if self.inplace:
      x.data, x.prev, x.grad_fn, x._backward = _apply(x.data), (self, ), "<TanhBackwards>", Backward.gelu_backwards(x, x)
      return x
    else:
      out = tensor(_apply(x.data), x.requires_grad, x.dtype)
      out.prev, out.grad_fn, out._backward = (x, ), "<TanhBackwards>", Backward.gelu_backwards(x, out)
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
      x.data, x.prev, x.grad_fn, x._backward = _apply(x.data), (self, ), "<LeakyReluBackwards>", Backward.lrelu_backwards(x, x)
      return x
    else:
      out = tensor(_apply(x.data), x.requires_grad, x.dtype)
      out.prev, out.grad_fn, out._backward = (self, ), "<LeakyReluBackwards>", Backward.lrelu_backwards(x, out)
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
      x.data, x.prev, x.grad_fn, x._backward = _apply(x.data), (self, ), "<SiluBackwards>", Backward.silu_backwards(x, x)
      return x
    else:
      out = tensor(_apply(x.data), requires_grad=True, dtype='float32')
      out.prev, out.grad_fn, out._backward = (self, ), "<SiluBackwards>", Backward.silu_backwards(x, out)
      return out

  def __call__(self, x:tensor):
    x = x if isinstance(x, tensor) else tensor(x, requires_grad=True, dtype='float32')
    return self.forward(x)
  
  def __repr__(self):
    return f"<LeakyRelu(inplace={self.inplace})>"