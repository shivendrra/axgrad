import math

def relu(x:float) -> float:
  return max(0, x)

def relu_derviative(x:float) -> float:
  return 1 if x > 0 else 0

def LeakyRelu(x:float, aplha:float=0.003) -> float:
  return 1 if x >= 0 else aplha * x

def LeakyRelu_derivative(x:float, aplha:float=0.003) -> float:
  return 1 if x > 0 else aplha

def tanh(x:float) -> float:
  return math.tanh(x)