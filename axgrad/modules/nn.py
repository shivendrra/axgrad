import random
from ..arrays import tensor
from .statics import ones, zeros

class Module:
  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0
  
  def parameters(self):
    return []

class Linear(Module):
  def __init__(self, _in, _out, bias: bool=False):
    self._in = _in
    self._out = _out
    self.weight = [[random.random() for _ in range(_in)] for _ in range(_out)]
    self.bias = [random.random() for _ in range(_out)] if bias is True else zeros([_out])
  
  def __call__(self, x):
    x = x if isinstance(x, tensor) else tensor(x)
    out = x * tensor(self.weight).transpose()
    return out

  def parameters(self):
    return self.weight + [self.bias]
  
  def __repr__(self):
    return f"Linear Layer({len(self.weight[0]) + len(self.weight[1])})"

class Sequence(Module):
  def __init__(self, _in, _out, **kwargs):
    self.neurons = [Linear(_in, _out, **kwargs) for _ in range(_out)]
  
  def __call__(self, x):
    out = [n(x) for n in self.neurons]
    return out[0] if len(out) == 1 else out
  
  def parameters(self):
    return [p for n in self.neurons for p in n.parameters()]

class MLP(Module):
  def __init__(self, _in, _out):
    self.layers = [Linear(_in, _out) for i in range(_out)]

  def __call__(self, x):
    for layers in self.layers:
      x = layers(x)
    return x
  
  def parameters(self):
    return [p for layers in self.layers for p in layers.parameters()]
  
  def __repr__(self):
    return f"MLP Layer"

class LayerNorm:
  def __init__(self, epsilon=1e-5):
    self.epsilon = epsilon
    self.mean = None
    self.variance = None

  def normalize(self, x):
    self.mean = sum(x) / len(x)
    self.variance = sum((xi - self.mean) ** 2 for xi in x) / len(x)
    normalized_x = [(xi - self.mean) / (self.variance + self.epsilon) ** 0.5 for xi in x]
    return normalized_x

  def denormalize(self, normalized_x):
    denormalized_x = [(xi * (self.variance + self.epsilon) ** 0.5) + self.mean for xi in normalized_x]
    return denormalized_x