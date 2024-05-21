class backward:
  @staticmethod
  def add_backward(one:list, two:list, out:list) -> None:
    def _backward():
      def accumulate_grad(grad, out):
        if isinstance(grad, list):
          for g, og in zip(grad, out):
            accumulate_grad(g, og)
          return
        grad += out
        print("grad: ", grad)
    
      accumulate_grad(one.grad, out.grad)
      accumulate_grad(two.grad, out.grad)
    return _backward
  
  @staticmethod
  def mul_backward(one:list, two:list, out:list) -> None:
    def _backward():
      def accumulate_grad(grad, out_grad, multiplier):
        if isinstance(grad, list):
          for g, og, m in zip(grad, out_grad, multiplier):
            accumulate_grad(g, og, m)
          return
        grad += out_grad * multiplier
    
      accumulate_grad(one.grad, out.grad, two.data)
      accumulate_grad(two.grad, out.grad, one.data)
    return _backward

  @staticmethod
  def matmul_backward(one, two, out):
    pass

  @staticmethod
  def pow_backward(one:list, out:list, exp:float) -> None:
    def _backward():
      def accumulate_grad(grad, out, exp):
        if isinstance(grad, list):
          for g, o in zip(grad, out):
            accumulate_grad(g, o, exp)
          return
        grad += (exp * out**(exp - 1)) * out
      
      accumulate_grad(one.grad, out.grad, exp)
    return _backward

  @staticmethod
  def relu_backward(one:list, out:list) -> None:
    def _backward():
      def accumulate_grad(grad, out_grad, data):
        if isinstance(grad, list):
          for g, og, d in zip(grad, out_grad, data):
            accumulate_grad(g, og, d)
          return
        grad += (data > 0) * out_grad
        print(grad)
      
      accumulate_grad(one.grad, out.grad, one.data)
    return _backward
  
  @staticmethod
  def tanh_backward(one:list, out:list) -> None:
    def _backward():
      def accumulate_grad(grad, out_grad, data):
        if isinstance(grad, list):
          for g, og, d in zip(grad, out_grad, data):
            accumulate_grad(g, og, d)
          return
        grad += (1 - data ** 2) * out_grad
        print(grad)
      
      accumulate_grad(one.grad, out.grad, one.data)
    return _backward
  
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