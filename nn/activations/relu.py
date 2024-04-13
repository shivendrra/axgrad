import math

class ReLU:
  @staticmethod
  def relu(x):
    return max(0, x)

  @staticmethod
  def relu_derivative(x):
    return 1 if x > 0 else 0

class LeakyRELU:
  @staticmethod
  def leaky_relu(x, alpha=0.01):
    return x if x >= 0 else alpha * x
  
  @staticmethod
  def leaky_relu_derivative(x, alpha=0.01):
    return 1 if x > 0 else alpha

class tanh:
  @staticmethod
  def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

  @staticmethod
  def tanh_derivative(x):
    return 1 - (tanh.tanh(x)**2)

class sigmoid:
  @staticmethod
  def sigmoid(x):
    return 1 / (1 + math.exp(-x))
  @staticmethod
  def sigmoid_derivative(x):
    return sigmoid.sigmoid(x)(1-sigmoid.sigmoid(x))
  
class softmax:
  @staticmethod
  def softmax(x):
    max_x = max(x)
    exp_values = [math.exp(i - max_x) for i in x]
    exp_sum = sum(exp_values)
    return [exp_value / exp_sum for exp_value in exp_values]