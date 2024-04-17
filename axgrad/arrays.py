class tensor:
  """ simple tensor function for axon"""
  
  def __init__(self, *args):
    self.data = args[0] if len(args) == 1 and isinstance(args[0], list) else list(args)
    self.shape = self.shape()

  def __repr__(self):
    return f"axon.tensor({self.data})"

  def __getitem__(self, index):
    return self.data[index]

  def __setitem__(self, index, value):
    self.data[index] = value

  def __add__(self, other):
    return tensor(self._operate(arr1=self.data, arr2=other.data, _op='+'))

  def __sub__(self, other):
    return tensor(self._operate(arr1=self.data, arr2=other.data, _op='-'))

  def __mul__(self, other):
    if isinstance(other, tensor) and self.data[1] == other.data[0]:
      return tensor([x * y for x, y in zip(self.data, other.data)])
    else:
      return tensor([x * other.data for x in self.data])

  def __truediv__(self, other):
    if isinstance(other, tensor):
      return tensor([x / y for x, y in zip(self.data, other.data)])
    else:
      return tensor([x / other for x in self.data])
  
  def shape(self):
    return tuple(self.get_shape(self.data))
  @staticmethod
  def _operate(arr1, arr2, _op=''):
    if len(arr1) != len(arr2):
      raise ValueError("Arrays must be of same shape & size")
    result = []
    for i in range(len(arr1)):
      result.append(tensor._operate(arr1[i], arr2[i], _op=_op)) if isinstance(arr1[i], list) and isinstance(arr2[i], list) else result.append(arr1[i] + arr2[i]) if _op=='+' else result.append(arr1[i] - arr2[i])
    return result

  @staticmethod
  def get_shape(arr):
    return [] if not isinstance(arr, list) else [len(arr)] + tensor.get_shape(arr[0])