from typing import *
from ._loss import MAE, MSE, HuberLoss
from .._tensor import tensor

class functional:
  def cross_entropy(outputs:Union[tensor, list], truths:Union[tensor, list]) -> tensor:
    pass

  def mse(outputs:Union[tensor, list], truths:Union[tensor, list]) -> tensor:
    loss = MSE(outputs, truths)
    return loss()

  def mae(outputs:Union[tensor, list], truths:Union[tensor, list]) -> tensor:
    loss = MAE(outputs, truths)
    return loss()
  
  def huber(outputs:Union[tensor, list], truths:Union[tensor, list]) -> tensor:
    loss = HuberLoss(outputs, truths)
    return loss()

  def softmax(data, dim:int):
    pass