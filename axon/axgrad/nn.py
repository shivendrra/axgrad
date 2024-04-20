from .modules.shape import get_shape, matmul, transpose
from .engine import Value
import random

class Module:
  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0
  
  def parameters(self):
    return []

class Linear(Module):
  def __init__(self, _in, nonlin):
    self.wei = [Value(random.uniform(-1, 1)) for _ in range(_in)]
    self.b = Value(0)
    self.nonlin = nonlin
  
  def __call__(self, x):
    act = sum((wi * xi for wi, xi in zip(self.wei, x)), self.b)
    return act.relu() if self.nonlin else act

  def parameters(self):
    return self.wei + [self.b]
  
  def __repr__(self) -> str:
    return f"Linear Neuron({len(self.wei)})"

class Linear2d(Module):
  """
    Linear layer similar to that of pytorch's nn.Linear
    - randomly initializes the weights and bias
    - wrapper over Value from micrograd

        `out = x * wT + b`
    
    returns:
      parameters [list]: to Module class
      out [list]: linearized outputs
  """
  def __init__(self, _in: int, _out: int, bias: bool =False) -> None:
    self.wei = [[Value(random.uniform(-1, 1)).data for _ in range(_in)] for _ in range(_out)]
    self.b = [Value(0).data for _ in range(_out)] if bias else None
  
  def __call__(self, x: list) -> Value:
    if get_shape(x) == get_shape(self.wei):
      out = matmul(x, transpose(self.wei))
      out = out + self.b if self.b is not None else out
    else:
      raise ArithmeticError(f"tensor shape error!")
    return out
  
  def parameters(self):
    return self.wei + self.b if self.b is not None else self.wei
  
  def __repr__(self) -> str:
    return f"2d Linear Neuron({len(self.wei)})"

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