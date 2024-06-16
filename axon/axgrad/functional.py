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
  def gelu_back(one:list, out:list):
    _back = geluBackward(one, out)
    return _back
  
  @staticmethod
  def leaky_r_backward(one:list, out:list):
    _back = leakyreluBackward(one, out)
    return _back
  
  @staticmethod
  def pow_back(one:list, out:list, exp:list):
    _back = PowBackward(one, out, exp)
    return _back
  
  # @staticmethod
  # def matmul_back(one:list, two:list, out:list):
  #   _back = MatMulBackward(one, two, out)
  #   return _back
  
  @staticmethod
  def sum_back(input_tensor, axis, keepdim, out):
    _back = SumBackward(input_tensor, axis, keepdim, out)
    return _back
  
  @staticmethod
  def trans_back(one:list, dim0:int, dim1:int, out:list):
    _back = TransposeBackward(one, dim0, dim1, out)
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