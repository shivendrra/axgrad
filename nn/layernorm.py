class LayerNormalization:
  def __init__(self, epsilon=1e-5):
    self.epsilon = epsilon
    self.mean = None
    self.variance = None

  def normalize(self, x):
    self.mean = sum(x) / len(x)
    self.variance = sum((xi - self.mean) ** 2 for xi in x) / len(x)
    normalized_x = [(xi - self.mean) / (self.variance + self.epsilon) ** 0.5 for xi in x]
    return normalized_x

  def denormalize(self, normalized_x):
    denormalized_x = [(xi * (self.variance + self.epsilon) ** 0.5) + self.mean for xi in normalized_x]
    return denormalized_x