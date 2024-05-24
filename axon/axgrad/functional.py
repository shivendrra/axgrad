from .functions import *
class backward:
  
  @staticmethod
  def add_back(one:list, two:list, out:list):
    _back = AddBackward(one, two, out)
    return _back
  
  @staticmethod
  def mul_back(one:list, two:list, out:list):
    _back = MulBackward(one, two, out)
    return _back
  
  @staticmethod
  def relu_back(one:list, out:list):
    _back = ReluBackward(one, out)
    return _back

  @staticmethod
  def tanh_back(one:list, out:list):
    _back = tanhBackward(one, out)
    return _back
  
  @staticmethod
  def sigmoid_back(one:list, out:list):
    _back = sigmoidBackward(one, out)
    return _back
  
  @staticmethod
  def pow_back(one:list, out:list, exp:list):
    _back = PowBackward(one, out, exp)
    return _back
  
  @staticmethod
  def backward(arr):
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v.prev:
          build_topo(child)
        topo.append(v)
    build_topo(arr)
    return topo