from typing import Any, Optional, List, Union
from ._tensor import tensor
from .ops.functionals import *

class ReLU:
  def __init__(self) -> None: pass

  def forward(self, x:Union[tensor, list]) -> tensor:
    def _apply(data):
      return [_apply(d) for d in data] if isinstance(data, list) else relu(data)
    return _apply(x)
  
  def backward(self, x):
    def _apply(data):
      return [_apply(d) for d in data] if isinstance(data, list) else relu_derivative(data)
    return _apply(x)
  
  def __call__(self, x:tensor) -> tensor:
    return self.forward(x)

class GELU:
  def __init__(self) -> None: pass

  def forward(self, x:Union[tensor, list]) -> tensor:
    def _apply(data):
      return [_apply(d) for d in data] if isinstance(data, list) else gelu(data)
    return _apply(x)
  
  def backward(self, x):
    def _apply(data):
      return [_apply(d) for d in data] if isinstance(data, list) else gelu_derivative(data)
    return _apply(x)
  
  def __call__(self, x:tensor) -> tensor:
    return self.forward(x)

class Sigmoid:
  def __init__(self) -> None: pass

  def forward(self, x:Union[tensor, list]) -> tensor:
    def _apply(data):
      return [_apply(d) for d in data] if isinstance(data, list) else sigmoid(data)
    return _apply(x)
  
  def backward(self, x):
    def _apply(data):
      return [_apply(d) for d in data] if isinstance(data, list) else sigmoid_derivative(data)
    return _apply(x)
  
  def __call__(self, x:tensor) -> tensor:
    return self.forward(x)