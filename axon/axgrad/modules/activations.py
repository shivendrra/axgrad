from ..engine import Value

class ReLU:
  def __call__(self, x) -> Value:
    for el in range(len(x)):
      x[el] = x[el].relu() if isinstance(x[el], Value) else Value(x[el]).relu()
    return x

class Tanh:
  def __call__(self, x) -> Value:
    for el in range(len(x)):
      x[el] = x[el].tanh() if isinstance(x[el], Value) else Value(x[el]).tanh()
    return x

class Sigmoid:
  def __call__(self, x) -> Value:
    for el in range(len(x)):
      x[el] = x[el].sigmoid() if isinstance(x[el], Value) else Value(x[el]).sigmoid()
    return x