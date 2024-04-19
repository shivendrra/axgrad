import math

class Value:
  def __init__(self, data, children=(), _op=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(children)
    self._op = _op

  def __repr__(self):
    return f"Value(data={self.data}, grad={self.grad})" if self.grad != '0' else f"Value(data={self.data})"

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    
    def _backward():
      self.grad += out.grad
      other.grad += out.grad
    out._backward = _backward
    return out
  
  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')
    
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out
  
  def __pow__(self, other):
    assert isinstance(other, (int, float))
    out = Value(self.data**other, (self,), f'**{other}')
    
    def _backward():
      self.grad += (other * self.data**(other-1)) * out.grad
    out._backward = _backward
    return out

  def tanh(self):
    t = (math.exp(2*self.data) -1)/(math.exp(2*self.data) + 1)
    out = Value(t, (self,), 'tanh')
    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    return out
  
  def sigmoid(self):
    t = 1 / (1 + math.exp(-self.data))
    out = Value(t, (self, ), 'sigmoid')
    def _backward():
      self.grad += t * (1 - t) * out.grad
    out._backward = _backward
    return out

  def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
    
    def _backward():
      self.grad += (out.data > 0) * out.grad
    out._backward = _backward
    return out

  def backward(self):
    """
      topoplogical order of all the children in graph
      and then apply chain rule for gradients, one at a time
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

    self.grad = 1
    for v in reversed(topo):
      v._backward()
  
  def __neg__(self):
    return self * -1  # -self

  def __radd__(self, other):
    return self + other  # other + self

  def __sub__(self, other):
    return self + (-other) # self - other

  def __rsub__(self, other):
    return other + (-self) # other - self

  def __rmul__(self, other):
    return self * other  # other * self

  def __truediv__(self, other):
    return self * other**-1  # self / other

  def __rtruediv__(self, other):
    return other * self**-1  # other / self