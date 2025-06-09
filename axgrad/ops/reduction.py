from ..helpers.ops import sum_axis, sum_axis0, mean_axis, mean_axis0, var_axis, var_axis0
from ..helpers.shape import flatten

def sum_tensor_ops(self, axis: int = None, keepdim: bool = False):
  from .._core import _tensor
  if axis is None:
    total = sum(flatten(self.data))
    out = [[total]] if keepdim else [total]
  elif axis == 0:
    out = sum_axis0(self.data)
  else:
    out = sum_axis(self.data, axis, keepdim)
  return _tensor(out, self.dtype)

def mean_tensor_ops(self, axis: int = None, keepdim: bool = False):
  from .._core import _tensor
  if axis is None:
    total = sum(flatten(self.data)) / self.size
    out = [[total]] if keepdim else [total]
  elif axis == 0:
    out = mean_axis0(self.data)
  else:
    out = mean_axis(self.data, axis, keepdim)
  return _tensor(out, self.dtype)

def var_tensor_ops(self, axis: int = None, ddof: int = 0, keepdim: bool = False):
  from .._core import _tensor
  if axis is None:
    flat = flatten(self.data)
    mean_val = sum(flat) / len(flat)
    variance = sum((x - mean_val) ** 2 for x in flat) / (len(flat) - ddof)
    out = [[variance]] if keepdim else [variance]
  elif axis == 0:
    out = var_axis0(self.data, ddof=ddof)
  else:
    mean_vals = mean_axis(self.data, axis, keepdim=False)
    out = var_axis(self.data, mean_vals, axis, ddof, keepdim)
  return _tensor(out, self.dtype)

def clip_tensor_ops(self, _min: float , _max: float):
  from .._core import _tensor
  def _clip(data):
    if isinstance(data, list):
      return [_clip(d, _min, _max) for d in data]
    return max(min(data, _max), _min)
  return _tensor(_clip(self.data), self.dtype)

def register_reduction_operators():
  from .._core import _tensor
  _tensor.sum = sum_tensor_ops
  _tensor.mean = mean_tensor_ops
  _tensor.var = var_tensor_ops
  _tensor.clip = clip_tensor_ops