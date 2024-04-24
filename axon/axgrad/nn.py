from .modules.shape import matmul, transpose
from .engine import Value
import random
from .modules.activations import ReLU

class Module:
  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0
  
  def parameters(self):
    return []

  def children(self):
    for attr_name in dir(self):
      attr = getattr(self, attr_name)
      if isinstance(attr, Module):
        yield attr

  def train(self, mode: bool=False):
    if not isinstance(mode, bool):
      raise ValueError("training mode is expected to be boolean")
    self.training = mode
    for modules in self.children():
      modules.train(mode)
    return self
  
  def eval(self):
    """ sets training to False """
    return self.train(False)

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

class Linear2d:
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
    self.wei = [[Value(random.uniform(-1, 1)) for _ in range(_in)] for _ in range(_out)]
    self.b = [Value(0) for _ in range(_out)] if bias else None
  
  def __call__(self, x: list) -> Value:
    if len(x[0]) == len(self.wei[0]):
      out = matmul(x, transpose(self.wei))
      if self.b is not None:
        out = [[out[i][j] + self.b[j] for j in range(len(out[0]))] for i in range(len(out))]
    else:
      raise ArithmeticError(f"tensor shape error!", len(x[0]), '!=', len(self.wei[0]))
    return out
  
  def parameters(self):
    params = []
    for row in self.wei:
      params.extend(row)
    if self.b is not None:
      params.extend(self.b)
    return params

class FeedForward(Module):
  """
    simple feedforward layer
    - two linear layers, one input & one output
    - relu as activation function
    
    returns:
      parameters [list]: to Module class
      out [list]: outputs of dim (_out, 1)
  """
  def __init__(self, _in, _out):
    self.layer1 = Linear2d(_in, _out, bias=False)
    self.relu = ReLU()
    self.layer2 = Linear2d(_out, 1, bias=False)
  
  def __call__(self, x):
    x = self.layer1(x)
    x = self.relu(x)
    x = self.layer2(x)
    return x

  def parameters(self):
    params = []
    for layer_params in [self.layer1.parameters(), self.layer2.parameters()]:
      params.extend(layer_params)
    return params

  def __repr__(self) -> str:
    return f"FeedForward"