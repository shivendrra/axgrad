import math

def relu(x:float) -> float:
  return max(0, x)

def relu_derivative(x:float) -> float:
  return 1 if x > 0 else 0

def LeakyRELU(x:float, alpha: float=0.03) -> float:
  return x if x >= 0 else alpha * x

def LeakyRELU_derivative(x:float, alpha:float= 0.03) -> float:
  return 1 if x > 0 else alpha

def tanh(x:float) -> float:
  return math.tanh(x)

def tanh_derivative(x:float) -> float:
  return 1 - (tanh(x)**2)

def sigmoid(x:float) -> float:
  return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x:float) -> float:
  return sigmoid(x)(1 - sigmoid(x))

def cdf(x:float) -> float:
  return 0.5 * (1 + tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x ** 3)))
  
def pdf(x:float) -> float:
  return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x**2)

def gelu(x:float) -> float:
  return x * cdf(x)

def gelu_derivative(x:float) -> float:
  return cdf(x) + x * pdf(x)