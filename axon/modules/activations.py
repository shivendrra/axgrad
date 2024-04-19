import math

def relu(x):
  return max(0, x)

def relu_derivative(x):
  return 1 if x > 0 else 0

def LeakyRELU(x, alpha: float=0.03):
  return x if x >= 0 else alpha * x

def LeakyRELU_derivative(x, alpha: float= 0.03):
  return 1 if x > 0 else alpha

def tanh(x):
  return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def tanh_derivative(x):
  return 1 - (tanh(x)**2)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
  return sigmoid(x)(1 - sigmoid(x))

def softmax(x):
  max_x = max(x)
  exp_values = [math.exp(i - max_x) for i in x]
  exp_sum = sum(exp_values)
  return [exp_value / exp_sum for exp_value in exp_values]