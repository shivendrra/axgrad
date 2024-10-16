from typing import *
from ._loss import MAE, MSE
from .._tensor import tensor

class functional:
  def __init__(self) -> None: pass

  def cross_entropy(self, outputs:Union[tensor, list], truths:Union[tensor, list]) -> tensor:
    pass

  def mse(self, outputs:Union[tensor, list], truths:Union[tensor, list]) -> tensor:
    loss = MSE(outputs, truths)
    return loss()

  def mae(self, outputs:Union[tensor, list], truths:Union[tensor, list]) -> tensor:
    loss = MAE(outputs, truths)
    return loss()

  def softmax(self, data, dim:int):
    pass