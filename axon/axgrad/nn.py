import random
from .engine import Value

class Module:
  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0
  
  def parameters(self):
    return []

class Linear(Module):
  def __init__(self, _in, nonlin):
    self.w = [Value(random.uniform(-1, 1)) for _ in range(_in)]
    self.b = Value(0)
    self.nonlin = nonlin
  
  def __call__(self, x):
    act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
    return act.relu() if self.nonlin else act

  def parameters(self):
    return self.w + [self.b]
  
  def __repr__(self) -> str:
    return f"Linear Neuron({len(self.w)})"

class Layer(Module):
  def __init__(self, n_in, n_out, **kwargs):
    self.neurons = [Linear(n_in, **kwargs) for _ in range(n_out)]
  
  def __call__(self, x):
    out = [n(x) for n in self.neurons]
    return out[0] if len(out) == 1 else out
  
  def parameters(self):
    return [p for n in self.neurons for p in n.parameters()]
  
  def __repr__(self) -> str:
    return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
  def __init__(self, n_in, n_out):
    sz = [n_in] + n_out
    self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(n_out)-1) for i in range(len(n_out))]

  def __call__(self, x):
    for layers in self.layers:
      x = layers(x)
    return x
  
  def parameters(self):
    return [p for layers in self.layers for p in layers.parameters()]

  def __repr__(self):
    return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"