from .modules.statics import get_shape, _operate, zeros, ones

class tensor:
  def __init__(self, *args, children=(), _op=''):
    self.data = args[0] if len(args) == 1 and isinstance(args[0], list) else list(args)
    self.shape = self.shape()
    self.grad = zeros(self.shape, dtype=float)
    self._backward = lambda: None
    self._prev = set(children)
    self._op = _op

  def __repr__(self):
    data_str = '\n\t'.join([str(row) for row in self.data])
    grad_str = '\n\t'.join([str(row) for row in self.grad])
    return f"axon.tensor(data={data_str},\ngrad={grad_str})\n"

  def __add__(self, other):
    """
      matrix addition of each element
    
      args:
      - self (tensor): first tensor
      - other (tensor): second tensor
    
      returns:
      - matrix with added corresponding values
    """
    other = other if isinstance(other, tensor) else tensor(other)
    out = tensor(_operate(arr1=self.data, arr2=other.data, op='+'), children=(self, other), _op='+')
    def _backward():
      def change(arr):
        for i in range(len(arr.data)):
          for j in range(len(arr.data[i])):
            arr.data[i][j] += out.grad[i][j]
          return arr
      self.grad = change(self)
      other.grad = change(other)
    out._backward = _backward
    return out

  def __sub__(self, other):
    """
      matrix subtraction of each element
    
      args:
      - self (tensor): first tensor
      - other (tensor): second tensor
    
      returns:
      - matrix with subtracted corresponding values
    """
    other = other if isinstance(other, tensor) else tensor(other)
    out = tensor(_operate(arr1=self.data, arr2=other.data, _op='-'), children=(self, other), _op='-')
    def _backward():
      def change(arr):
        for i in range(len(arr.data)):
          for j in range(len(arr.data[i])):
            arr.data[i][j] += out.grad[i][j]
        return arr
      self.grad = change(self)
      other.grad = change(other)
    out._backward = _backward
    return out

  def __mul__(self, other):
    """
      matrix multiplication for self, and other as a & b
    
      args:
      - self (tensor): first tensor
      - other (tensor): second tensor
    
      returns:
      - multiplied matrix with shape (len(a[0]), len(b(1)))
    """
    if self.shape[1] != other.shape[0]:
      raise ValueError(f"invalid shape for matrix multiplication: {self.shape} != {other.shape}")
    else:
      out = tensor([[sum(self.data[i][k] * other.data[k][j] for k in range(len(other.data))) for j in range(len(other.data[0]))] for i in range(len(self.data))], children=(self, other), _op='*')
      def _backward():
        def change(arr, trg):
          for i in range(len(arr.data)):
            for j in range(len(arr.data[i])):
              arr.data[i][j] += out.grad[i][j] * trg.data[i][j]
          return arr
        self.grad = change(self, other)
        other.grad = change(other, self)
        out._backward = _backward
      return out

  def __truediv__(self, other):
    raise ArithmeticError(f"can't do a matrix division")
  
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
    out = tensor([[self.data[i][j]**other for j in range(len(self.data[i]))] for i in range(len(self.data))], children=(self,), _op=f'**{other}')
    def _backward():
      for i in range(len(self.data)):
        for j in range(len(self.data[i])):
          self.grad += (other * self.data[i][j]**(other-1)) * out.grad
    out._backward = _backward
    return out

  def relu(self):
    """
      simple relu function with backward pass on it's derivative
    
      args:
      - self (tensor): first tensor
    
      returns:
      - multiplied matrix with elements x if (x > 0)
    """
    out = tensor([[max(0, self.data[i][j]) for j in range(len(self.data[i]))] for i in range(len(self.data))])
    def _backward():
      def change(arr):
        for i in range(len(arr.data)):
          for j in range(len(arr.data[i])):
            arr.data[i][j] += out.grad[i][j] * (out.data[i][j] > 0)
        return arr
      self.grad = change(self)
    out._backward = _backward
    return out

  def backward(self):
    """
      calculates the gradient for all the items in _prev w.r.t final matrix

      args:
      - self (tensor): final tensor that contains all the operated features

      returns:
      - self.grad (tensor): updated gradients on the matrix
    """
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad = ones(self.shape, dtype=float)
    for v in reversed(topo):
      v._backward()

  def shape(self):
    """
      computes the shape of a tensor

      returns:
        tuple: tuple that contains the shape of the tensor
    """
    return tuple(get_shape(self.data))

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