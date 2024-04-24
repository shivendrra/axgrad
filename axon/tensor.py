from helpers.shape import get_shape, check_arr
from helpers.statics import zeros, ones

def _operate(arr1, arr2, op=''):
  """
    staticmethod to carry addition or subtraction for __add__ & __sub__

    args:
      - arr1: first tensor
      - arr2: second tensor
      - _op: '+' or '-'
    
    returns:
      - matrix with performed operation
  """
  if len(arr1) != len(arr2):
    raise ValueError("Arrays must be of same shape & size")
  result = []
  for i in range(len(arr1)):
    if isinstance(arr1[i], list) and isinstance(arr2[i], list):
      result.append(_operate(arr1[i], arr2[i], op=op))
    else:
      if op=='+':
        result.append(arr1[i] + arr2[i])
      elif op=='-':
        result.append(arr1[i] - arr2[i])
      elif op=='*':
        result.append(arr1[i] * arr2[i])
      elif op=='/':
        result.append(arr1[i] / arr2[i])
    return result
  
  else:
    raise ArithmeticError(f"both matrix should be of same shape for element level {op}")

def matmul_2d(arr1, arr2):
  if check_arr(arr1, arr2) is True:
    out = [[sum(arr1[i][k] * arr2[k][j] for k in range(len(arr2))) for j in range(len(arr2[0]))] for i in range(len(arr1))]
    return out
  else:
    raise ArithmeticError(f"Shape are not matching: {len(arr1[0])} should be equal to {len(arr2[1])}")

class tensor:
  def __init__(self, *data, requires_grad:bool=False, children:set=()) -> None:
    self.data = data[0] if len(data) == 1 and isinstance(data[0], list) else list(data)
    self.shape = self.shape()
    self.grad = zeros(self.shape, dtype=float)
    self._prev = set(children)
    self.req_grad = requires_grad

  def __repr__(self):
    data_str = '\n\t'.join([str(row) for row in self.data])
    grad_str = '\n\t'.join([str(row) for row in self.grad])
    return f"axon.tensor(data={data_str},\ngrad={grad_str})\n" if self.req_grad is True else f"axon.tensor({data_str})"
  
  def __add__(self, other):
    other = other if isinstance(other, tensor) else tensor(other)
    out = tensor(_operate(arr1=self.data, arr2=other.data, op='+'), requires_grad=self.req_grad, children=(self, other))
    return out
  
  def __mul__(self, other):
    other = other if isinstance(other, tensor) else tensor(other)
    out = tensor(_operate(arr1=self.data, arr2=other.data, op='*'), requires_grad=self.req_grad, children=(self, other))
    return out
  
  def __sub__(self, other):
    other = other if isinstance(other, tensor) else tensor(other)
    out = tensor(_operate(arr1=self.data, arr2=other.data, op='-'), requires_grad=self.req_grad, children=(self, other))
    return out
  
  def __truediv__(self, other):
    other = other if isinstance(other, tensor) else tensor(other)
    out = tensor(_operate(arr1=self.data, arr2=other.data, op='/'), requires_grad=self.req_grad, children=(self, other))
    return out

  def shape(self):
    return get_shape(self.data)