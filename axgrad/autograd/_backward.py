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
    _back = TanhBackward(one, out)
    return _back
  
  @staticmethod
  def sigmoid_back(one:list, out:list):
    _back = SigmoidBackward(one, out)
    return _back
  
  @staticmethod
  def gelu_back(one:list, out:list):
    _back = GELUBackward(one, out)
    return _back
  
  @staticmethod
  def leaky_r_back(one:list, out:list):
    _back = LeakyRELUBackward(one, out)
    return _back
  
  @staticmethod
  def silu_back(one:list, out:list):
    _back = SiluBackward(one, out)
    return _back
  
  @staticmethod
  def pow_back(one:list, out:list, exp:list):
    _back = PowBackward(one, out, exp)
    return _back
  
  @staticmethod
  def matmul_back(one:list, two:list, out:list):
    _back = MatMulBackward(one, two, out)
    return _back
  
  @staticmethod
  def sum_back(one:list, out:list):
    _back = SumBackward(one, out)
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