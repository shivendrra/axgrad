from .helpers.shape import get_shape, broadcast_array, broadcast_shapes
from .helpers.statics import zeros, ones

def check_arr(arr1, arr2):
  if get_shape(arr1.data)[1] == get_shape(arr2.data)[0] and len(get_shape(arr2.data)) == 2:
    return True
  else:
    return False
  
def _ops_unpack(arr):
  new = []
  if isinstance(arr, list):
    for i in arr:
      if isinstance(i, list):
        new.extend(_ops_unpack(i))
      elif isinstance(i, int) or isinstance(i, float):
        new.append(arr)
        break
  return new

def _unpack(arr, new=None):
  if new is None:
    new = []
  if isinstance(arr, list):
    for i in arr:
      _unpack(i, new)
  elif isinstance(arr, int) or isinstance(arr, float):
    new.append(arr)
  return new

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
  
  def __getitem__(self, index):
    return self.data[index]
  
  def __setitem__(self, index, value):
    self.data[index] = value
  
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
    out = zeros(self.shape)
    for i in range(len(self.data)):
      for j in range(len(self.data[i])):
        out[i][j] = self.data[i][j] + other.data[i][j]
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
    if self.shape != other.shape:
      raise ValueError(f"Arrays must be of same shape & size {self.shape} != {other.shape}")
    else:
      out = zeros(self.shape)
      self = tensor(_ops_unpack(self))
      other = tensor(_ops_unpack(other))

      for i in range(len(self.data)):
        for j in range(len(self.data[i])):
          out[i][j] = self.data[i][j] * other.data[i][j]
    # out = tensor(_operate(arr1=self.data, arr2=other.data, op='*'), requires_grad=self.req_grad, child=(self, other))
    return tensor(out)
  
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
    out = zeros(self.shape)
    for i in range(len(self.data)):
      for j in range(len(self.data[i])):
        out[i][j] = self.data[i][j] - other.data[i][j]
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
    out = zeros(self.shape)
    for i in range(len(self.data)):
      for j in range(len(self.data[i])):
        out[i][j] = self.data[i][j] - other.data[i][j]
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
  
  def unpack(self):
    """
      unpacks a n-dim tensor into a list

    Args:
      self (tensor): n-dim tensor to unpack

    Returns:
      new (list): 1-dim list with all the elements
    """
    new = _unpack(self.data)
    return new

  def _sum(self):
    """
      unpacks the n-dim tensor & then sums up all the elements
      into a single integer/float

    Args:
      self (tensor): n-dim tensor to sum

    Returns:
      sum (int/float): sum of all the elements present in the n-dim tensor
    """
    unpacked_arr = _unpack(self.data)
    return sum(i for i in unpacked_arr)
  
  def broadcast(self, other):
    other = other if isinstance(other, tensor) else tensor(other)
    new_shape = broadcast_shapes(self.shape, other.shape)
    self_broadcasted = broadcast_array(self.data, new_shape)
    other_broadcasted = broadcast_array(other.data, new_shape)
    return tensor(self_broadcasted), tensor(other_broadcasted)

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

  @staticmethod
  def matmul(x, y):
    x = x if isinstance(x, tensor) else tensor(x)
    y = y if isinstance(y, tensor) else tensor(y)
    if len(x.data[0]) != len(y.data):
      raise ValueError("Matrices have incompatible dimensions for multiplication.")

    out = zeros((len(x.data), len(y.data[0])))
    y_t = y.transpose().data
    for i in range(len(x.data)):
      for j in range(len(y_t)):
        out[i][j] = sum(x.data[i][k] * y_t[j][k] for k in range(len(y.data)))
    return tensor(out)