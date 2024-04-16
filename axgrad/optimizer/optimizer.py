from typing import Any
from ..engine import Value
from .loss import Loss
from ..nn_mods import Module

class Optim:
  class sgd:
    def __init__(self, params, learn_rate, momentum=0.0, acceleration=0.0):
      self.velo = [Value(0.0) for _ in params]
      self.lr = learn_rate
      self.moment = momentum
      self.acc = acceleration
      self.params = params
    
    def step(self):
      for p, v in zip(self.params, self.velo):
        p.grad.backward()
        v.data = v.data * self.moment + p.grad.data
        p.data = p.data - v.data * self.lr

      return self.step