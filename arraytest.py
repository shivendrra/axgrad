from axon.axgrad import Value
import random

class Linear2d:
  def __init__(self, _in: int, _out: int, bias: bool =False) -> None:
    self.wei = [[Value(random.uniform(-1, 1)).data for _ in range(_in)] for _ in range(_out)]
    self.b = [Value(0).data for _ in range(_out)] if bias else None
  
  def __call__(self, x: list) -> Value:
    pass


"""
  Linear: y = x * wei^T + b
"""

linear = Linear2d(2, 4, bias=True)
print(linear.wei)
print(linear.b)