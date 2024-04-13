class array:
  """ simple array function for axgrad"""
  
  def __init__(self, *args):
    if len(args) == 1 and isinstance(args[0], list):
      self.data = args[0]
      self.shape = (len(args[0]),)
    else:
      self.data = list(args)
      self.shape = (len(args),)

  def __repr__(self):
    return f"array({self.data})"

  def __getitem__(self, index):
    return self.data[index]

  def __setitem__(self, index, value):
    self.data[index] = value

  def __add__(self, other):
    if isinstance(other, array):
      return array([x + y for x, y in zip(self.data, other.data)])
    else:
      return array([x + other for x in self.data])

  def __sub__(self, other):
    if isinstance(other, array):
      return array([x - y for x, y in zip(self.data, other.data)])
    else:
      return array([x - other for x in self.data])

  def __mul__(self, other):
    if isinstance(other, array):
      return array([x * y for x, y in zip(self.data, other.data)])
    else:
      return array([x * other for x in self.data])

  def __truediv__(self, other):
    if isinstance(other, array):
      return array([x / y for x, y in zip(self.data, other.data)])
    else:
      return array([x / other for x in self.data])