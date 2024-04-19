from ..arrays import tensor
from ..statics.statics import zeros, ones
import numpy as np

class Sequential:
  def __init__(self, *modules):
    self.modules = modules

  def __call__(self, x):
    for module in self.modules:
      x = module(x)
    return x if isinstance(x, tensor) else tensor(x)

class LayerNorm:
  def __init__(self, n_features, eps=1e-5):
    self.n_features = n_features
    self.eps = eps
    self.gamma = ones(n_features)
    self.beta = zeros(n_features)

  def __call__(self, x):
    mean = np.mean(x, dim=-1, keepdim=True)
    std = np.std(x, dim=-1, keepdim=True)
    x_normalized = (x - mean) / (std + self.eps)
    y = self.gamma * x_normalized + self.beta
    return y

class Dropout:
  def __init__(self, dropout_prob):
    self.dropout_prob = dropout_prob
    self.mask = None
            
  def _call_(self, x, training=True):
    if training:
      self.mask = (np.random.rand(*x.shape) < self.dropout_prob) / self.dropout_prob
      return x * self.mask
    else:
      return x

class Softmax:
  def __call__(self, x, axis=-1):
    x -= np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    softmax_probs = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    return softmax_probs