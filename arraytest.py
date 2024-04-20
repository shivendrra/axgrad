from axon.axgrad import Value
import random

def get_shape(arr):
  if isinstance(arr, list):
    return (len(arr), ) + get_shape(arr[0])
  else:
    return ()

def matmul(arr1, arr2):
  xs, ys = get_shape(arr1), get_shape(arr2)
  if xs[1] == ys[0]:
    out = [[sum(arr1[i][k] * arr2[k][j] for k in range(len(arr2))) for j in range(len(arr2[0]))] for i in range(len(arr1))]
    
    del xs, ys
    return out
  else:
    raise ArithmeticError(f"tensor shape error! : {xs[0]} != {ys[1]}")

def transpose(arr):
  R, C = get_shape(arr)
  return [[arr[i][j] for i in range(R)] for j in range(C)]

class ReLU:
  """
    applies relu activation to the list items
      `y = x if x > 0 else 0`
    
    returns:
      x [list]: containing new non-linear values
  """
  def __call__(self, x) -> Value:
    return [[xi.relu().data for xi in row] for row in x]

class Module:
  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0
  
  def parameters(self):
    return []

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
    return self.wei + self.b if self.b is not None else self.wei

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
    self.layer1 = Linear2d(_in, _out, bias=True)
    self.relu = ReLU()
    self.layer2 = Linear2d(_out, 1, bias=False)
  
  def __call__(self, x):
    x = self.layer1(x)
    x = self.relu(x)
    x = self.layer2(x)
    return x

  def parameters(self):
    return self.layer1.parameters() + self.layer2.parameters()
  
  def __repr__(self) -> str:
    return f"FeedForward"

class Loss:

  def error(self, trg, prd):
    """
      simple difference function

      Args:
        trg (int or floar): ground truth
        prd (int or float): logits

      Returns:
        int or float: difference b/w logits & ground truth
    """
    return prd - trg

  def mse_loss(self, trg, prd):
    """
      simple mean squared error loss function
        
        'mse_loss = sum(error**2)/len(trg)'

      Args:
        trg (list): list containing target tokens
        prd (list): list containing logits

      Returns:
        axgrad.Value: float value of loss as Value object for backprop
    """
    loss = sum(self.error(ygt, yout)**2 for ygt, yout in zip(trg, prd)) / len(trg)
    return loss if isinstance(loss, Value) else Value(loss)

xs = [
  [-2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, -1.0, 1.0],
  [1.0, 1.0, -1.0],
]

ys = [-1.0, 1.0, -1.0, 1.0]
n = FeedForward(3, 4)

for k in range(10):
  ypred = n(xs)
  loss = Value(sum((ygt - yout[0].data)**2 for ygt, yout in zip(ys, ypred)) / len(ys))
  n.zero_grad()
  loss.backward()

for p in n.parameters():
  print(p)

# loss = loss_f.mse_loss(ys, ypred)
# print(loss)