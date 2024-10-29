from .._tensor import tensor
from ._parameters import Parameter
from ._module import Module
from .._ops import conv2d
from ..helpers.utils import _randn, _zeros
from ..autograd._backward import Backward

class Linear(Module):
  def __init__(self, _in, _out, bias=False):
    super(Linear, self).__init__()
    self.wei = Parameter(_randn(shape=(_in, _out)))
    self.bias = Parameter(_zeros(shape=(1, _out))) if bias else None
  def __repr__(self): return f"<LinearLayer in_features={self.wei.shape[0]} out_features={self.wei.shape[1]}>"
  def __call__(self, x): return self.forward(x)
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

class Conv2d(Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    super(Conv2d, self).__init__()
    self.kernel = Parameter(_randn(shape=(out_channels, in_channels)))
    self.stride, self.padding = stride, padding
    self.bias = Parameter(_zeros(shape=(out_channels, 1))) if bias else None
  def __call__(self, x): return self.forward(x)
  def __repr__(self): return f"<Conv2dLayer in_channels={self.kernel.shape[1]} out_channels={self.kernel.shape[0]} kernel_size={self.kernel.shape[2:]}>"
  @staticmethod
  def _pair(value): return (value, value) if isinstance(value, int) else value
  def forward(self, x):
    x = x if isinstance(x, tensor) else tensor(x, dtype=tensor.float32, requires_grad=True)
    out = conv2d(x, self.kernel)
    if self.bias is not None:
      out = out + self.bias
    return out

  def parameters(self):
    params = [self.kernel]
    if self.bias is not None:
      params.append(self.bias)
    return params

class Embedding(Module):
  def __init__(self, num_embeddings:int, embedding_dim:int):
    super().__init__()
    self.weight = tensor(_randn(shape=(num_embeddings, embedding_dim)), requires_grad=True, dtype="float32")
  def __call__(self, x): return self.forward(x)
  def __repr__(self): return f"<EmbeddingLayer n_embeddings={self.weight.shape[0]} embedding_dim={self.weight.shape[1]}>"
  def forward(self, indices):
    indices = indices.F.data if hasattr(indices, 'F') else indices

    out = [self.weight[i] for i in indices]
    out = tensor(out, requires_grad=True)
    out.prev, out.grad_fn, out._backward = (self.weight,), "<EmbeddingBackwards>", Backward.embed_backwards(self.weight, indices, out)
    return out