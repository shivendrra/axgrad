from .._tensor import tensor
from ._parameters import Parameter
from ._module import Module
from ..helpers.utils import _randn, _zeros, _ones

class LayerNorm(Module):
  def __init__(self, normalized_shape: tuple, eps=1e-5, elementwise_affine=True, bias=False):
    super(LayerNorm, self).__init__()
    self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    self.eps = eps
    self.elementwise_affine = elementwise_affine

    if self.elementwise_affine:
      self.gamma = Parameter(_randn(shape=self.normalized_shape))
      self.beta = Parameter(_zeros(shape=self.normalized_shape))
    else:
      self.gamma = None
      self.beta = None

    self.bias = Parameter(_zeros(shape=self.normalized_shape)) if bias else None

  def __call__(self, x):
    return self.forward(x) + self.bias if self.bias is not None else self.forward(x)

  def forward(self, x):
    x = x if isinstance(x, tensor) else tensor(x, dtype=tensor.float32, requires_grad=True)

    mean = x.mean(axis=-1, keepdims=True).unsqueeze(dim=0)
    var = x.var(axis=-1, keepdims=True).unsqueeze(dim=0)
    x_norm = (x - mean) / (var + [self.eps]).sqrt()

    if self.elementwise_affine:
      gamma = self.gamma.reshape((1,) * (x.ndim - len(self.gamma.shape)) + tuple(self.gamma.shape))
      beta = self.beta.reshape((1,) * (x.ndim - len(self.beta.shape)) + tuple(self.beta.shape))
      x_norm = x_norm * gamma + beta

      # Ensure to attach backward functions to the parameters for gradient calculation
      self.gamma.grad = (x_norm * (x - mean) / (var + [self.eps]).sqrt()).sum(axis=0)
      self.beta.grad = x_norm.sum(axis=0)
    x_norm.grad_fn = "<LayerNorm>"
    return x_norm

  def parameters(self):
    params = []
    if self.elementwise_affine:
      params.extend([self.gamma, self.beta])
    return params

  def __repr__(self):
    return f"<LayerNorm normalized_shape={self.normalized_shape}, eps={self.eps}>"

class BatchNorm(Module):
  def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
    super(BatchNorm, self).__init__()
    self.num_features, self.eps, self.momentum, self.affine = num_features, eps, momentum, affine
    self.track_running_stats = track_running_stats

    if self.affine:
      self.gamma = Parameter(_randn(shape=(1, num_features)))
      self.beta = Parameter(_zeros(shape=(1, num_features)))
    else:
      self.gamma = None
      self.beta = None

    if self.track_running_stats:
      self.running_mean = _zeros(shape=(1, num_features))
      self.running_var = _ones(shape=(1, num_features))
    else:
      self.running_mean = None
      self.running_var = None

  def __call__(self, x):
    return self.forward(x)

  def forward(self, x):
    x = x if isinstance(x, tensor) else tensor(x, dtype=tensor.float32, requires_grad=True)

    if self.training:
      batch_mean = x.mean(axis=0, keepdims=True)
      batch_var = ((x - batch_mean) ** 2).mean(axis=0, keepdims=True)
      if self.track_running_stats:
        self.running_mean = self.momentum * batch_mean + (1 - self.momentum) * self.running_mean
        self.running_var = self.momentum * batch_var + (1 - self.momentum) * self.running_var
      mean, var = batch_mean, batch_var
    else:
      mean, var = self.running_mean, self.running_var
    x_norm = (x - mean) / (var + self.eps) ** 0.5
    if self.affine:
      x_norm = x_norm * self.gamma + self.beta
    return x_norm

  def parameters(self):
    params = []
    if self.affine:
      params.extend([self.gamma, self.beta])
    return params

  def __repr__(self):
    return f"<BatchNorm num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}>"