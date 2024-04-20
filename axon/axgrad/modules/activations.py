from ..engine import Value

class ReLU:
  """
    applies relu activation to the list items
      `y = x if x > 0 else 0`
    
    returns:
      x [list]: containing new non-linear values
  """
  def __call__(self, x) -> Value:
    return [[xi.relu() for xi in row] for row in x]

class Tanh:
  """
    applied tanh activation to each element in list
      `y = (e^2x - 1)/(e^2x + 1)`
    
    returns:
      x [list]: containing new non-linear values
  """
  def __call__(self, x) -> Value:
    return [[xi.tanh() for xi in row] for row in x]

class Sigmoid:
  """
    applies sigmoid activation to each element in list
      `y = 1 / (1 + exp^-x)`
    
    returns:
      x [list]: containing new non-linear values
  """
  def __call__(self, x) -> Value:
    return [[xi.sigmoid() for xi in row] for row in x]