from ...helpers.utils import zeros

class Optim:
  class SGD:
    def __init__(self, params, lr:float=0.1, momentum:float=0, acceleration:float=0) -> None:
      self.velo = zeros((1, len(params)))
      self.lr = lr
      self.moment = momentum
      self.acc = acceleration
      self.params = params

    def step(self):
      for p, v in zip(self.params, self.velo):
        v.data = v.data * self.moment + p.grad
        p.data += -self.lr * v.data + self.acc * (v.data - p.grad)
      return self.step

  class AdamW:
    def __init__(self, params, lr:float=0.01) -> None:
      self.params = params
      self.lr = lr

    def step(self):
      return self.step