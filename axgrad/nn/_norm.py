from .._tensor import tensor
from ._parameters import Parameter
from ._module import Module
from ..helpers.utils import _randn, _zeros, _ones
from ..autograd._backward import Backward

class LayerNorm(Module):
  def __init__(self, normalized_shape: tuple, eps=1e-5, elementwise_affine=True, bias=False):
    super(LayerNorm, self).__init__()
    self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    self.eps, self.elementwise_affine = eps, elementwise_affine

    self.gamma = Parameter(_randn(shape=self.normalized_shape)) if self.elementwise_affine else None
    self.beta = Parameter(_zeros(shape=self.normalized_shape)) if self.elementwise_affine else None
    self.bias = Parameter(_zeros(shape=self.normalized_shape)) if bias else None

  def __call__(self, x): return self.forward(x) + self.bias if self.bias is not None else self.forward(x)
  def __repr__(self): return f"<LayerNorm normalized_shape={self.normalized_shape}, eps={self.eps}>"

  def forward(self, x):
    x = x if isinstance(x, tensor) else tensor(x, dtype=tensor.float32, requires_grad=True)

    mean = x.mean(axis=-1, keepdims=True).unsqueeze(dim=0)
    var = x.var(axis=-1, keepdims=True).unsqueeze(dim=0)
    out = (x - mean) / (var + [self.eps]).sqrt()

    if self.elementwise_affine:
      gamma = self.gamma.reshape((1,) * (x.ndim - len(self.gamma.shape)) + tuple(self.gamma.shape))
      beta = self.beta.reshape((1,) * (x.ndim - len(self.beta.shape)) + tuple(self.beta.shape))
      out = out * gamma + beta
    out.prev, out.grad_fn, out._backward = (self.gamma, self.beta), "<LayerNormBackwards>", Backward.layernorm_backwards(self.gamma, self.beta, self.bias, out, self.elementwise_affine, self.eps, x, mean, var)
    return out

  def parameters(self):
    params = []
    if self.elementwise_affine:
      params.extend([self.gamma, self.beta])
    if self.bias is not None:
      params.append([self.bias])
    return params

class BatchNorm(Module):
  def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
    super(BatchNorm, self).__init__()
    self.num_features, self.eps, self.momentum, self.affine = num_features, eps, momentum, affine
    self.track_running_stats = track_running_stats

    self.gamma = Parameter(_randn(shape=(1, num_features))) if self.affine else None
    self.beta = Parameter(_zeros(shape=(1, num_features))) if self.affine else None

    self.running_mean = tensor(_zeros(shape=(1, num_features))) if self.track_running_stats else None
    self.running_var = tensor(_ones(shape=(1, num_features))) if self.track_running_stats else None
  
  def __call__(self, x): return self.forward(x)
  def __repr__(self): return f"<BatchNorm num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}>"

  def forward(self, x):
    x = x if isinstance(x, tensor) else tensor(x, dtype=tensor.float32, requires_grad=True)

    if self.training:
      batch_mean = x.mean(axis=0, keepdims=True)
      batch_var = ((x - batch_mean) ** 2).mean(axis=0, keepdims=True)
      if self.track_running_stats:
        self.running_mean = [self.momentum] * batch_mean + [(1 - self.momentum)] * self.running_mean
        self.running_var = [self.momentum] * batch_var + [(1 - self.momentum)] * self.running_var
      mean, var = batch_mean, batch_var
    else:
      mean, var = self.running_mean, self.running_var
    out = (x - mean) / (var + [self.eps]).sqrt()
    if self.affine:
      out = out * self.gamma + self.beta
    out.prev, out.grad_fn, out._backward = (self.gamma, self.beta), "<BatchNormBackwards>", Backward.batchnorm_backwards(self.gamma, self.beta, self.running_mean, self.running_var, self.affine, out, self.eps, x, mean, var)
    return out

  def parameters(self):
    params = []
    if self.affine:
      params.extend([self.gamma, self.beta])
    return params