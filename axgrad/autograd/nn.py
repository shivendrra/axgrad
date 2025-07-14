from ..tensor import Tensor
from ..utils import zeros_like
from .._core import lib
from ctypes import c_float, c_int
import math

class EmbeddingBackwards:
  def __init__(self, weight, indices): 
    self.input = [weight, indices]
    self.weight_shape, self.weight_ndim, self.weight_size = weight.shape, weight.ndim, weight.size
    self.indices_shape, self.indices_ndim = indices.shape, indices.ndim

  def backward(self, grad):
    weight, indices = self.input
    if grad is None or grad.data is None: raise ValueError("Gradient cannot be None")
    weight_grad = zeros_like(weight)

    if indices.ndim == 0:
      idx = int(indices.item())
      if 0 <= idx < self.weight_shape[0]:
        for j in range(self.weight_shape[1]):
          grad_val = lib.get_item_tensor(grad.data, (c_int * 1)(j))
          indices_ctypes = (c_int * 2)(idx, j)
          lib.set_item_tensor(weight_grad.data, indices_ctypes, c_float(grad_val))
    else:
      flat_indices, flat_grad = indices.flatten(), grad.flatten()
      for i in range(flat_indices.size):
        idx = int(flat_indices[i])
        if 0 <= idx < self.weight_shape[0]:
          for j in range(self.weight_shape[1]):
            grad_idx = i * self.weight_shape[1] + j
            grad_indices_ctypes = (c_int * 1)(grad_idx)
            grad_val = lib.get_item_tensor(flat_grad.data, grad_indices_ctypes)
            weight_indices_ctypes = (c_int * 2)(idx, j)
            current_val = lib.get_item_tensor(weight_grad.data, weight_indices_ctypes)
            new_val = current_val + grad_val
            lib.set_item_tensor(weight_grad.data, weight_indices_ctypes, c_float(new_val))
    
    return [weight_grad, None]