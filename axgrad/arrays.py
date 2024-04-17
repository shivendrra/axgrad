from .modules.matrices import zeros

class tensor:
  def __init__(self, *args):
    self.data = args[0] if len(args) == 1 and isinstance(args[0], list) else list(args)
    self.shape = self.shape()

  def __repr__(self):
    return f"axon.tensor({self.data})"

  def __getitem__(self, index):
    return self.data[index]

  def __setitem__(self, index, value):
    self.data[index] = value

  def __add__(self, other):
    """
      matrix addition of each element
    
      args:
      - self: first tensor
      - other: second tensor
    
      returns:
      - matrix with added corresponding values
    """
    return tensor(self._operate(arr1=self.data, arr2=other.data, _op='+'))

  def __sub__(self, other):
    """
      matrix subtraction of each element
    
      args:
      - self: first tensor
      - other: second tensor
    
      returns:
      - matrix with subtracted corresponding values
    """
    return tensor(self._operate(arr1=self.data, arr2=other.data, _op='-'))

  def __mul__(self, other):
    """
      matrix multiplication for self, and other as a & b
    
      args:
      - self: first tensor
      - other: second tensor
    
      returns:
      - multiplied matrix with shape (len(a[0]), len(b(1)))
    """
    if len(self.data[0]) != len(other.data):
      raise ValueError(f"invalid shape for matrix multiplication: {self.shape} != {other.shape}")
    else:
      return tensor([[sum(self.data[i][k] * other.data[k][j] for k in range(len(other.data))) for j in range(len(other.data[0]))] for i in range(len(self.data))])

  def __truediv__(self, other):
    """
      matrix multiplication for matrix1 with matrix2's transpose
    
      args:
      - self: first tensor
      - other: second tensor
    
      returns:
      - multiplied matrix
    """
    if isinstance(other, tensor):
      if len(self.data[0]) != len(other.data):
        raise ValueError(f"invalid shape for matrix multiplication: {self.shape} != {other.shape}")
    else:
      trans_mat = other.data.transpose()
      return tensor([[sum(self.data[i][k] * other.data[k][j] for k in range(len(other.data))) for j in range(len(other.data[0]))] for i in range(len(self.data))])
  
  def shape(self):
    return tuple(self.get_shape(self.data))
  
  @staticmethod
  def _operate(arr1, arr2, _op=''):
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
      result.append(tensor._operate(arr1[i], arr2[i], _op=_op)) if isinstance(arr1[i], list) and isinstance(arr2[i], list) else result.append(arr1[i] + arr2[i]) if _op=='+' else result.append(arr1[i] - arr2[i])
    return result

  @staticmethod
  def get_shape(arr):
    """
      args:
      - arr: array for determining the shape
  
      returns:
      - tuple with the shape
    """
    return [] if not isinstance(arr, list) else [len(arr)] + tensor.get_shape(arr[0])
  
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