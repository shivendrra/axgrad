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

  def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
    def _backward():
      self.grad += (out.data > 0) * out.grad
    out._backward = _backward
    return out

  def backward(self):
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
    return self * -1

  def __radd__(self, other):
    return self + other

  def __sub__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data - other.data, (self, other), '-')
    def _backward():
      self.grad += out.grad
      other.grad += out.grad
    out._backward = _backward
    return out

  def __rsub__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(other.data - self.data, (self, other), '-')
    def _backward():
      self.grad += out.grad
      other.grad += out.grad
    out._backward = _backward
    return out

  def __rmul__(self, other):
    return self * other

  def __truediv__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data**-1, (self, other), '/')
    def _backward():
      self.grad += (-other.data**-2) * out.grad
      other.grad += (-self.data**-2) * out.grad
    out._backward = _backward
    return out

  def __rtruediv__(self, other):
    return other * self**-1