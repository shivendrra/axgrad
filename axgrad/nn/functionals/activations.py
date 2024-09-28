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
      out = tensor(_apply(self.data), self.requires_grad, self.dtype)
      out.prev, out.grad_fn, out._backward = (self, ), "<ReluBackward>", Backward.relu_backwards(self, out)
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
      out = tensor(_apply(self.data), self.requires_grad, self.dtype)
      out.prev, out.grad_fn, out._backward = (self, ), "<TanhBackward>", Backward.tanh_backwards(self, out)
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
      out = tensor(_apply(self.data), self.requires_grad, self.dtype)
      out.prev, out.grad_fn, out._backward = (self, ), "<SigmoidBackward>", Backward.sigmoid_backwards(self, out)
      return out

  def __call__(self, x:tensor):
    x = x if isinstance(x, tensor) else tensor(x, requires_grad=True, dtype='float32')
    return self.forward(x)
  
  def __repr__(self):
    return f"<Sigmoid(inplace={self.inplace})>"

class ReLU(Module):
  def __init__(self, inplace=False):
    super().__init__()
    self.inplace = inplace

  def forward(self, x:tensor):
    def _apply(data):
      return [_apply(d) for d in data] if isinstance(data, list) else gelu(data)
    if self.inplace:
      x.data, x.prev, x.grad_fn, x._backward = _apply(x.data), (self, ), "<ReluBackwards>", Backward.gelu_backwards(x, x)
      return x
    else:
      out = tensor(_apply(self.data), self.requires_grad, self.dtype)
      out.prev, out.grad_fn, out._backward = (self, ), "<GeluBackward>", Backward.gelu_backwards(self, out)
      return out

  def __call__(self, x:tensor):
    x = x if isinstance(x, tensor) else tensor(x, requires_grad=True, dtype='float32')
    return self.forward(x)
  
  def __repr__(self):
    return f"<Gelu(inplace={self.inplace})>"

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
      x._backward = Backward.lrelu_backwards(x, x)
      return x
    else:
      out = tensor(_apply(x.data), requires_grad=True, dtype='float32')
      out.prev = (self, )
      out.grad_fn = "<LeakyReluBackward>"
      out._backward = Backward.lrelu_backwards(x, out)
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
      x._backward = Backward.silu_backwards(x, x)
      return x
    else:
      out = tensor(_apply(x.data), requires_grad=True, dtype='float32')
      out.prev = (self, )
      out.grad_fn = "<SiluBackward>"
      out._backward = Backward.silu_backwards(x, out)
      return out

  def __call__(self, x:tensor):
    x = x if isinstance(x, tensor) else tensor(x, requires_grad=True, dtype='float32')
    return self.forward(x)
  
  def __repr__(self):
    return f"<LeakyRelu(inplace={self.inplace})>"