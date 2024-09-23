import math
from typing import *

def relu(x:Union[float, int]) -> Union[float, int]:
  return max(x, 0)

def relu_derivative(x:Union[float, int]) -> Union[float, int]:
  return 1 if x > 0 else 0

def LeakyRelu(x:Union[float, int], alpha:float = 1e-3) -> Union[float, int]:
  return x if x >= 0 else x*alpha

def LeakyRelu_derivative(x:Union[float, int], alpha:float = 1e-3) -> Union[float, int]:
  return 1 if x > 0 else alpha

def tanh(x:Union[float, int]) -> Union[float, int]:
  return math.tanh(x)

def tanh_derivative(x:Union[float, int]) -> Union[float, int]:
  return 1 - (tanh(x)**2)

def sigmoid(x:Union[float, int]) -> Union[float, int]:
  return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x:Union[float, int]) -> Union[float, int]:
  return sigmoid(x) * (1 - sigmoid(x))

def cdf(x:Union[float, int]) -> Union[float, int]:
  return 0.5 * (1 + tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x ** 3)))

def pdf(x:Union[float, int]) -> Union[float, int]:
  return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x ** 2)

def gelu(x:Union[float, int]) -> Union[float, int]:
  return x * cdf(x)

def gelu_derivative(x:Union[float, int]) -> Union[float, int]:
  return cdf(x) + x * pdf(x)

def silu(x:Union[float, int]) -> Union[float, int]:
  return x * sigmoid(x)

def silu_derivative(x:Union[float, int]) -> Union[float, int]:
  return silu(x) + sigmoid(x) * (1-silu(x))