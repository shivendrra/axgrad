from ..engine import Value

class Optim:
  class sgd:
    """
      stochastic gradient descent algorithm
      args:
        params: model parameters
        learn_rate: learning rate for descent
        momentum: momentum for the descent
        acceleration: acceleration for the descent
      function:
        step: iterates over each parameter and updates the corresponding gradients and velocity
              along with the momentum and the finally updates the parameters
    """
    def __init__(self, params, learn_rate=0.01, momentum=0, acceleration=0):
      self.velo = [Value(0.0) for _ in params]
      self.lr = learn_rate
      self.moment = momentum
      self.acc = acceleration
      self.params = params

    def step(self):
      for p, v in zip(self.params, self.velo):
        v.data = v.data * self.moment + p.grad
        p.data += -self.lr * v.data + self.acc * (v.data - p.grad)
      return self.step

  class gd:
    """
      < doesn't work properly has some issues >
      
      simple gradient descent algorithm
      args:
        params: model parameters
        learn_rate: learning rate for descent
        reg_f: regularization factor
      function:
        step: iterates over each parameter and updates the corresponding gradients while applying
              regulariztion factor
    """
    def __init__(self, params, learn_rate, reg_f=0.0):
      self.lr = learn_rate
      self.reg = reg_f
      self.params = params

    def step(self):
      for p in self.params:
        p.data += -self.lr * (p.grad + self.reg * p.data)
      return self.step