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
    return out
  else:
    raise ArithmeticError(f"tensor shape error! : {xs[0]} != {ys[1]}")

def transpose(arr):
  R, C = get_shape(arr)
  return [[arr[i][j] for i in range(R)] for j in range(C)]

class Module:
  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0
  
  def parameters(self):
    return []

class Linear2d:
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

x = [[Value(random.uniform(-1, 1)).data for _ in range(2)] for _ in range(4)]

linear = Linear2d(2, 4, bias=True)
li = linear(x)
print(linear.parameters())