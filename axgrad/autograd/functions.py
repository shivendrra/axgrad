import math

class AddBackwards:
  def __init__(self, x, y): self.input = [x, y]
  def backward(self, grad): return [grad, grad]

class SubBackwards:
  def __init__(self, x, y): self.input = [x, y]
  def backward(self, grad): return [grad, -grad]

class MulBackwards:
  def __init__(self, x, y): self.input = [x, y]
  def backward(self, grad): return [self.input[0] * grad, self.input[1] * grad]

class DivBackwards:
  def __init__(self, x, y): self.input = [x, y]
  def backward(self, grad): return [grad * (self.input[1] ** -1), grad * (-self.input[0] / (self.input[1] ** 2))]

class PowBackwards:
  def __init__(self, x, exp): self.input = [x, exp]
  def backward(self, grad):
    base, exp = self.input[0], self.input[1]
    if isinstance(base, (int, float)): g_base, g_exp = grad * (base ** (exp - 1)), (grad * base ** exp) * math.log(base)
    else: g_base, g_exp = grad * exp * (base ** (exp - 1)), (grad * base ** exp) * (base.log())
    return [g_base, g_exp]

class LogBackward:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad / self.input[0]]

class ExpBackward:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad * self.input[0].exp()]

class SqrtBackward:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad * (0.5 / self.input[0].sqrt())]

class MatmulBackward:
  def __init__(self, x, y): self.input = [x, y]
  def backward(self, grad): return [grad @ self.input[1].transpose(), self.input[0].transpose() @ grad]

class SinBackward:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad * self.input.cos()]

class SinhBackward:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad * self.input.cosh()]

class CosBackward:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad * (-self.input.sin())]

class CoshBackward:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad * self.input.sinh()]

class TanBackward:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad * (self.input.cos() ** -2)]

class TanhBackward:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad * (1 - self.input.tanh() ** 2)]

class TransposeBackward:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad.transpose()]

class FlatBackward:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad.flatten()]

class ReshapeBackward:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad.reshape(self.input[0].shape)]