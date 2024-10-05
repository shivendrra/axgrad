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

class HuberLoss:
  def __init__(self, outputs: Union[tensor, list], truth: Union[tensor, list], delta: float = 1.0) -> None:self.outputs, self.truth, self.delta = outputs, truth, delta
  def __call__(self) -> tensor[float]:
    ## loss += 0.5 * r^2
    ## loss += delta * (|r| - 0.5 * delta)
    ## loss += loss / total_elements
    diff = self.truth - self.outputs
    abs_diff = absolute(diff)
    squared_loss = 0.5 * (diff ** 2)
    linear_loss = self.delta * (abs_diff - 0.5 * self.delta)

    loss_data = []
    for i in range(len(abs_diff)):
      if abs_diff[i] <= self.delta:
        loss_data.append(squared_loss[i])
      else:
        loss_data.append(linear_loss[i])

    total_loss = sum(loss_data)
    return total_loss / tensor([self.truth.numel], requires_grad=True, dtype=self.truth.dtype)