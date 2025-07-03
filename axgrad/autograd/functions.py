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
  def backward(self, grad): return [grad * (1 - self.output ** 2)]

class SigmoidBackwards:
  def __init__(self, x, out): self.input, self.output = [x], out
  def backward(self, grad): return [grad * (self.output * (1 - self.output))]

class ReluBackwards:
  def __init__(self, x, out): self.input, self.output = [x], out
  def backward(self, grad):
    # ReLU derivative: 1 if x > 0, else 0
    mask = self.output > 0
    return [grad * mask]

class EluBackwards:
  def __init__(self, x, out): self.input, self.output = [x], out
  def backward(self, grad):
    # ELU derivative: 1 if x > 0, else alpha * exp(x) = alpha + f(x) if x <= 0
    # Since f(x) = alpha * (exp(x) - 1) for x <= 0, derivative is alpha * exp(x) = alpha + f(x)
    x = self.input[0]
    mask = x > 0
    # For x > 0: derivative is 1, for x <= 0: derivative is output + alpha (but alpha is small, approximating as output + 1e-5)
    derivative = mask * 1.0 + (1 - mask) * (self.output + 1e-5)
    return [grad * derivative]

class LeakyReluBackwards:
  def __init__(self, x, out): self.input, self.output = [x], out
  def backward(self, grad):
    # LeakyReLU derivative: 1 if x > 0, else eps (slope parameter)
    x = self.input[0]
    mask = x > 0
    # Default eps = 1e-5 from the forward function
    derivative = mask * 1.0 + (1 - mask) * 1e-5
    return [grad * derivative]

class GeluBackwards:
  def __init__(self, x, out): self.input, self.output = [x], out
  def backward(self, grad):
    # GELU derivative: 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))) + 
    #                  x * 0.5 * (1 - tanh^2(sqrt(2/π) * (x + 0.044715 * x^3))) * sqrt(2/π) * (1 + 3 * 0.044715 * x^2)
    x = self.input[0]
    sqrt_2_pi = (2.0 / math.pi) ** 0.5
    inner = sqrt_2_pi * (x + 0.044715 * x ** 3)
    tanh_inner = inner.tanh()
    sech2_inner = 1 - tanh_inner ** 2
    derivative = 0.5 * (1 + tanh_inner) + x * 0.5 * sech2_inner * sqrt_2_pi * (1 + 3 * 0.044715 * x ** 2)
    return [grad * derivative]

class SwishBackwards:
  def __init__(self, x, out): self.input, self.output = [x], out
  def backward(self, grad):
    # Swish derivative: beta * sigmoid(beta * x) + beta * x * sigmoid(beta * x) * (1 - sigmoid(beta * x))
    # Since output = x * sigmoid(beta * x), and default beta = 1e-5
    x = self.input[0]
    beta = 1e-5  # default beta from forward function
    sigmoid_val = (beta * x).sigmoid()
    derivative = beta * sigmoid_val + beta * x * sigmoid_val * (1 - sigmoid_val)
    return [grad * derivative]

class SiluBackwards:
  def __init__(self, x, out): self.input, self.output = [x], out
  def backward(self, grad):
    # SiLU (Swish with beta=1) derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    x = self.input[0]
    sigmoid_val = x.sigmoid()
    derivative = sigmoid_val + x * sigmoid_val * (1 - sigmoid_val)
    return [grad * derivative]

class SoftplusBackwards:
  def __init__(self, x, out): self.input, self.output = [x], out
  def backward(self, grad):
    return [grad * self.input[0].sigmoid()]

class TransposeBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad.transpose()]

class FlatBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad.reshape(self.input[0].shape)]

class ReshapeBackwards:
  def __init__(self, x): self.input = [x]
  def backward(self, grad): return [grad.reshape(self.input[0].shape)]