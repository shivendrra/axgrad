from .helpers.shape import get_shape
from .helpers.statics import zeros, ones

def check_arr(arr1, arr2):
  if get_shape(arr1.data)[1] == get_shape(arr2.data)[0] and len(get_shape(arr2.data)) == 2:
    return True
  else:
    return False

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

class tensor:
  def __init__(self, *data, requires_grad:bool=False, child:set=()) -> None:
    self.data = data[0] if len(data) == 1 and isinstance(data[0], list) else list(data)
    self.shape = self.shape()
    self.grad = zeros(self.shape, dtype=float)
    self._prev = set(child)
    self.req_grad = requires_grad

  def __repr__(self):
    data_str = '\n\t'.join([str(row) for row in self.data])
    grad_str = '\n\t'.join([str(row) for row in self.grad])
    return f"axon.tensor(data={data_str},\ngrad={grad_str})\n" if self.req_grad is True else f"axon.tensor({data_str})"
  
  def __add__(self, other):
    """
      matrix's element level addition
      both matrices should have same shape
    
      args:
      - self (tensor): first tensor
      - other (tensor): second tensor
    
      returns:
      - matrix with added corresponding values
    """
    other = other if isinstance(other, tensor) else tensor(other)
    out = tensor(_operate(arr1=self.data, arr2=other.data, op='+'), requires_grad=self.req_grad, children=(self, other))
    return out
  
  def __mul__(self, other):
    """
      matrix's element level multiplication
      both matrices should have same shape
    
      args:
      - self (tensor): first tensor
      - other (tensor): second tensor
    
      returns:
      - matrix with multiplied corresponding values
    """
    other = other if isinstance(other, tensor) else tensor(other)
    out = tensor(_operate(arr1=self.data, arr2=other.data, op='*'), requires_grad=self.req_grad, children=(self, other))
    return out
  
  def __sub__(self, other):
    """
      matrix's element level subtraction
      both matrices should have same shape
    
      args:
      - self (tensor): first tensor
      - other (tensor): second tensor
    
      returns:
      - matrix with subtracted corresponding values
    """
    other = other if isinstance(other, tensor) else tensor(other)
    out = tensor(_operate(arr1=self.data, arr2=other.data, op='-'), requires_grad=self.req_grad, children=(self, other))
    return out
  
  def __truediv__(self, other):
    """
      matrix's element level division
      both matrices should have same shape
    
      args:
      - self (tensor): first tensor
      - other (tensor): second tensor
    
      returns:
      - matrix with divided corresponding values
    """
    other = other if isinstance(other, tensor) else tensor(other)
    out = tensor(_operate(arr1=self.data, arr2=other.data, op='/'), requires_grad=self.req_grad, children=(self, other))
    return out
  
  def __pow__(self, other):
    """
      raises the power of the elements in a matrix
    
      args:
      - self (tensor): first tensor
      - other (int or float): power to be raised
    
      returns:
      - multiplied matrix of same shape as input matrix with each element's power being
        raised to other
    """
    assert isinstance(other, (int, float))
    raise NotImplementedError("__pow__ function not implemented")

  def shape(self):
    """
      computes the shape of a tensor

      returns:
        tuple: tuple that contains the shape of the tensor
    """
    return get_shape(self.data)
  
  def transpose(self):
    """
      args:
      - self: tensor to be transposed
    
      returns:
      - transposed matrix
    """
    rows = len(self.data)
    cols = len(self.data[0])
    return tensor([[self.data[i][j] for i in range(rows)] for j in range(cols)])
  
  @staticmethod
  def matmul_2d(arr1, arr2):

    if check_arr(arr1, arr2) is True:
      out = tensor([[sum(arr1.data[i][k] * arr2.data[k][j] for k in range(len(arr2.data))) for j in range(len(arr2.data[0]))] for i in range(len(arr1.data))], child=(arr1, arr2))
      return out
    else:
      raise ArithmeticError(f"Shape are not matching: {len(arr1[0])} should be equal to {len(arr2[1])}")

  @staticmethod
  def convolution_2d(image, kernel):
    img_h, img_w = len(image), len(image[0])
    ker_h, ker_w = len(kernel), len(kernel[0])

    output_height = img_h - ker_h + 1
    output_width = img_w - ker_w + 1

    output = [[0] * output_width for _ in range(output_height)]

    for i in range(output_height):
      for j in range(output_width):
        for k in range(ker_h):
          for l in range(ker_w):
            output[i][j] += image[i+k][j+l] * kernel[k][l]

    return output