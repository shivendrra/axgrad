"""
  @nn/_conv.py performs convolutions
  @brief inputs a tensor and performs convolutions on it by randomly initializing the kernel according to the size
  @comments:
  - none...
"""

from .._tensor import tensor
from ._parameters import Parameter
from ._module import Module
from ..helpers.utils import _randn, _zeros
from .._ops import conv2d

class Conv2d(Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    super(Conv2d, self).__init__()
    self.kernel = Parameter(_randn(shape=(out_channels, in_channels)))
    self.stride, self.padding = stride, padding
    if bias:
      self.bias = Parameter(_zeros(shape=(out_channels, 1)))
    else:
      self.bias = None

  def __call__(self, x):
    return self.forward(x)

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

  def __repr__(self):
    return f"<Conv2dLayer in_channels={self.kernel.shape[1]} out_channels={self.kernel.shape[0]} kernel_size={self.kernel.shape[2:]}>"

  @staticmethod
  def _pair(value):
    """Ensures the kernel size is always a tuple (height, width)."""
    if isinstance(value, int):
      return (value, value)
    return value