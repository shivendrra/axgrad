import math

class AddBackwards:
  def __init__(self, x, y): self.input = [x, y]
  def backward(self, grad): return [grad, grad]

class SubBackwards:
  def __init__(self, x, y): self.input = [x, y]
  def backward(self, grad): return [grad, -grad]

class MulBackwards:
  def __init__(self, x, y): self.input = [x, y]
  def backward(self, grad): return [self.input[1] * grad, self.input[0] * grad]

class DivBackwards:
  def __init__(self, x, y): self.input = [x, y]
  def backward(self, grad): return [grad * (self.input[1] ** -1), grad * (-self.input[0] / (self.input[1] ** 2))]

class NegBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad.__neg__()]

class PowBackwards:
  def __init__(self, x, exp): self.input = [x, exp]
  def backward(self, grad):
    base, exp = self.input[0], self.input[1]
    if isinstance(base, (int, float)): g_base, g_exp = grad * (base ** (exp - 1)), (grad * base ** exp) * math.log(base)
    else: g_base, g_exp = grad * exp * (base ** (exp - 1)), (grad * base ** exp) * (base.log())
    return [g_base, g_exp]

class RPowBackwards:
  def __init__(self, base, exp): self.input = [base, exp]
  def backward(self, grad): return [None, grad * (self.input[0] ** self.input[1]) * math.log(self.input[0])]

class LogBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad / self.input[0]]

class AbsBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad * self.input[0].sign()]

class ExpBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad * self.input[0].exp()]

class SqrtBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad * (0.5 / self.input[0].sqrt())]

class MatmulBackwards:
  def __init__(self, x, y): self.input = [x, y]
  def backward(self, grad): return [grad @ self.input[1].transpose(), self.input[0].transpose() @ grad]

class DotBackwards:
  def __init__(self, x, y): self.input = [x, y]
  def backward(self, grad): return [grad @ self.input[1].transpose(), self.input[0].transpose() @ grad]

class SinBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad * self.input[0].cos()]

class SinhBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad * self.input[0].cosh()]

class CosBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad * (-self.input[0].sin())]

class CoshBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad * self.input[0].sinh()]

class TanBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad * (self.input[0].cos() ** -2)]

class TanhBackwards:
  def __init__(self, x, out): self.input, self.output = [x], out
  def backward(self, grad): return [grad * (self.output ** 2  - 1).__neg__()]

class TransposeBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad.transpose()]

class FlatBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad.reshape(self.input[0].shape)]

class ReshapeBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad.reshape(self.input[0].shape)]