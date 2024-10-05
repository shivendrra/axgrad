"""
  @nn/_loss.py
  @brief file containing all the required loss functions
"""

from typing import *
from .._tensor import tensor

def absolute(value):
  return -value if value < 0 else value

class MSE:
  def __init__(self, outputs:Union[tensor, list], truth:Union[tensor, list]) -> None: self.truth, self.outputs = truth, outputs
  def __call__(self) -> tensor[float]:
    ## mse = sum((y - y') ** 2) / total_elements
    loss = ((self.truth - self.outputs) ** 2)
    loss = loss.sum() / tensor([self.truth.numel], requires_grad=True, dtype=self.truth.dtype)
    return loss

class MAE:
  def __init__(self, outputs:Union[tensor, list], truth:Union[tensor, list]) -> None: self.truth, self.outputs = truth, outputs
  def __call__(self) -> tensor[float]:
    ## mae = sum(|y - y'|) / total_elements
    loss = absolute((self.truth - self.outputs) ** 2)
    loss = loss.sum() / tensor([self.truth.numel], requires_grad=True, dtype=self.truth.dtype)
    return loss