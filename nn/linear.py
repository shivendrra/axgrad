import random

class LinearLayer:
  def __init__(self, in_features, out_features):
    self.in_features = in_features
    self.out_features = out_features
    self.weight = [[random.random() for _ in range(in_features)] for _ in range(out_features)]
    self.bias = [random.random() for _ in range(out_features)]

  def __call__(self, x):
    return self.forward(x)

  def forward(self, x):
    output = []
    for i in range(self.out_features):
      linear_sum = 0
      for j in range(self.in_features):
        linear_sum += x[j] * self.weight[i][j]
      output.append(linear_sum + self.bias[i])
    return output