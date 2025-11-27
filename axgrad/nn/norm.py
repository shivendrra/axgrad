from .module import Module
from .parameter import Parameter
from .._core import lib
from ..helpers import DtypeHelp
from ctypes import c_int, c_size_t
import math
from ..linalg.norm import *

class LayerNorm(Module):
  def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, dtype="float32"):
    super().__init__()
    if isinstance(normalized_shape, int): normalized_shape = (normalized_shape,)
    self.normalized_shape, self.eps, self.elementwise_affine, self.dtype = normalized_shape, eps, elementwise_affine, dtype
    if elementwise_affine:
      self.weight, self.bias = Parameter(normalized_shape, dtype), Parameter(normalized_shape, dtype)
      self._init_parameters()
    else: self.weight, self.bias = None, None

  def _init_parameters(self):
    ones_data = lib.ones_tensor((c_int * len(self.normalized_shape))(*self.normalized_shape), c_size_t(math.prod(self.normalized_shape)), c_size_t(len(self.normalized_shape)), c_int(DtypeHelp._parse_dtype(self.dtype))).contents
    self.weight.data = ones_data
    zeros_data = lib.zeros_tensor((c_int * len(self.normalized_shape))(*self.normalized_shape), c_size_t(math.prod(self.normalized_shape)), c_size_t(len(self.normalized_shape)), c_int(DtypeHelp._parse_dtype(self.dtype))).contents
    self.bias.data = zeros_data
  def forward(self, input):
    normalized = std_norm(input)
    return normalized * self.weight + self.bias if self.elementwise_affine else normalized
  def inner_repr(self): return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"

class BatchNorm1d(Module):
  def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, dtype="float32"):
    super().__init__()
    self.num_features, self.momentum, self.eps, self.affine, self.dtype = num_features, momentum, eps, affine, dtype
    if affine:
      self.weight, self.bias = Parameter((num_features,), dtype), Parameter((num_features,), dtype)
      self._init_parameters()
    else: self.weight, self.bias = None, None
  def _init_parameters(self):
    ones_data = lib.ones_tensor((c_int * 1)(self.num_features), c_size_t(self.num_features), c_size_t(1), c_int(DtypeHelp._parse_dtype(self.dtype))).contents
    self.weight.data = ones_data
    zeros_data = lib.zeros_tensor((c_int * 1)(self.num_features), c_size_t(self.num_features), c_size_t(1), c_int(DtypeHelp._parse_dtype(self.dtype))).contents
    self.bias.data = zeros_data
  def forward(self, input):
    normalized = std_norm(input)
    return normalized * self.weight + self.bias if self.affine else normalized
  def inner_repr(self): return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}"

class RMSNorm(Module):
  def __init__(self, normalized_shape, eps=1e-8, elementwise_affine=True, dtype="float32"):
    super().__init__()
    if isinstance(normalized_shape, int): normalized_shape = (normalized_shape,)
    self.normalized_shape, self.eps, self.elementwise_affine, self.dtype = normalized_shape, eps, elementwise_affine, dtype
    if elementwise_affine:
      self.weight = Parameter(normalized_shape, dtype)
      self._init_parameters()
    else: self.weight = None
  def _init_parameters(self):
    ones_data = lib.ones_tensor((c_int * len(self.normalized_shape))(*self.normalized_shape),  c_size_t(math.prod(self.normalized_shape)),  c_size_t(len(self.normalized_shape)),  c_int(DtypeHelp._parse_dtype(self.dtype))).contents
    self.weight.data = ones_data
  def forward(self, input):
    normalized = rms_norm(input)
    return normalized * self.weight if self.elementwise_affine else normalized
  def inner_repr(self): return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"

class GroupNorm(Module):
  def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, dtype="float32"):
    super().__init__()
    self.num_groups, self.num_channels, self.eps, self.affine, self.dtype = num_groups, num_channels, eps, affine, dtype
    if num_channels % num_groups != 0: raise ValueError(f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})")
    if affine:
      self.weight, self.bias = Parameter((num_channels,), dtype), Parameter((num_channels,), dtype)
      self._init_parameters()
    else: self.weight, self.bias = None, None
  def _init_parameters(self):
    ones_data = lib.ones_tensor((c_int * 1)(self.num_channels), c_size_t(self.num_channels), c_size_t(1), c_int(DtypeHelp._parse_dtype(self.dtype))).contents
    self.weight.data = ones_data
    zeros_data = lib.zeros_tensor((c_int * 1)(self.num_channels), c_size_t(self.num_channels), c_size_t(1), c_int(DtypeHelp._parse_dtype(self.dtype))).contents
    self.bias.data = zeros_data
  def forward(self, input):
    normalized = std_norm(input)
    if self.affine: return normalized * self.weight + self.bias
    return normalized
  def inner_repr(self): return f"{self.num_groups}, {self.num_channels}, eps={self.eps}, affine={self.affine}"

class InstanceNorm1d(Module):
  def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, dtype="float32"):
    super().__init__()
    self.num_features, self.eps, self.momentum, self.affine, self.dtype = num_features, eps, momentum, affine, dtype
    if affine:
      self.weight, self.bias = Parameter((num_features,), dtype), Parameter((num_features,), dtype)
      self._init_parameters()
    else: self.weight, self.bias = None, None
  def _init_parameters(self):
    ones_data = lib.ones_tensor((c_int * 1)(self.num_features), c_size_t(self.num_features), c_size_t(1), c_int(DtypeHelp._parse_dtype(self.dtype))).contents
    self.weight.data = ones_data
    zeros_data = lib.zeros_tensor((c_int * 1)(self.num_features), c_size_t(self.num_features), c_size_t(1), c_int(DtypeHelp._parse_dtype(self.dtype))).contents
    self.bias.data = zeros_data
  def forward(self, input):
    normalized = std_norm(input)
    if self.affine: return normalized * self.weight + self.bias
    return normalized
  def inner_repr(self): return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}"

class LocalResponseNorm(Module):
  def __init__(self, size, alpha=1e-4, beta=0.75, k=1.0, dtype="float32"):
    super().__init__()
    self.size, self.alpha, self.beta, self.k, self.dtype = size, alpha, beta, k, dtype
  def forward(self, input): return robust_norm(input)
  def inner_repr(self): return f"size={self.size}, alpha={self.alpha}, beta={self.beta}, k={self.k}"

class SpectralNorm(Module):
  def __init__(self, module, name='weight', n_power_iterations=1, eps=1e-12, dtype="float32"):
    super().__init__()
    self.module, self.name, self.n_power_iterations, self.eps, self.dtype = module, name, n_power_iterations, eps, dtype
  def forward(self, *args, **kwargs):
    weight = getattr(self.module, self.name)
    normalized_weight = unit_norm(weight)
    setattr(self.module, self.name, normalized_weight)
    return self.module(*args, **kwargs)
  def inner_repr(self): return f"name={self.name}, n_power_iterations={self.n_power_iterations}, eps={self.eps}"

class Clip(Module):
  def __init__(self, max_val, dtype="float32"):
    super().__init__()
    self.max_val, self.dtype = max_val, dtype
  def forward(self, input): return clip(input, self.max_val)
  def inner_repr(self): return f"max_val={self.max_val}"

class Clamp(Module):
  def __init__(self, min_val, max_val, dtype="float32"):
    super().__init__()
    self.min_val, self.max_val, self.dtype = min_val, max_val, dtype
  def forward(self, input): return clamp(input, self.min_val, self.max_val)
  def inner_repr(self): return f"min_val={self.min_val}, max_val={self.max_val}"