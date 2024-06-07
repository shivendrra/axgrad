from .helpers.utils import *
from .helpers.shape import *
from .tensor import tensor

def matmul(a, b):
  def _remul(a, b):
    if len(a.shape) == 2 and len(b.shape) == 2:
      out = zeros((len(a.data), len(b.data[0])))
      b_t = transpose(b.data)
      for i in range(len(a.data)):
        for j in range(len(b_t)):
          out[i][j] = sum(a.data[i][k] * b_t[j][k] for k in range(len(a.data[0])))
      return out
    else:
      out_shape = a.shape[:-1] + (b.shape[-1],)
      out = zeros(out_shape)
      for i in range(len(a.data)):
        out[i] = _remul(tensor(a.data[i]), tensor(b.data[i]))
      return out

  if a.shape[-1] != b.shape[-2]:
    raise ValueError("Matrices have incompatible dimensions for matmul")

  a = a if isinstance(a, tensor) else tensor(a)
  b = b if isinstance(b, tensor) else tensor(b)
  out = tensor(_remul(a, b), child=(a, b), _ops='<matmul>')
  # out._backward = backward.matmul_back(a, b, out)
  return out