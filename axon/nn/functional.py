from ..helpers.utils import zeros_like
from ..tensor import tensor
import math

def softmax(x, dim=None):
  if dim is not None and dim < 0:
    dim = x.ndim + dim
  
  x_max = x.max(axis=dim, keepdim=True)
  exp_x = math.e ** (x - x_max)

  if dim is not None:
    sum_exp_x = exp_x.sum(axis=dim, keepdim=True) + zeros_like(exp_x)
  else:
    sum_exp_x = exp_x.sum()

  return exp_x / sum_exp_x

def error(trg:tensor, prd:tensor) -> tensor:
  return trg - prd

def mse(trg:list, prd:list) -> tensor:
  trg = trg if isinstance(trg, tensor) else tensor(trg)
  prd = prd if isinstance(prd, tensor) else tensor(prd)
  if trg.shape == prd.shape:
    loss = error(trg, prd).sum(axis=0)
  else:
    raise ValueError(f"Predicted values should be equal to Ground Values: {trg.shape} != {prd.shape}")
  loss = tensor(loss, child=(trg, prd), _ops='<MseLoss>')
  return loss