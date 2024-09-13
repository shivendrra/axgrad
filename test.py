class tensor:
  def __init__(self, data, requires_grad=False, dtype=float):
    self.data = data
    self.shape = self.get_shape(data)
    self.requires_grad = requires_grad
    self.dtype = dtype
  
  def get_shape(self, data):
    if isinstance(data, list):
      return [len(data)] + self.get_shape(data[0]) if data else []
    return []

  def flatten(self):
    # Flatten the data for reshaping
    if isinstance(self.data, list):
      return [item for sublist in self.data for item in sublist]
    return [self.data]

  def view(self, *new_shape):
    flat_data = self.flatten()
    total_elements = len(flat_data)

    if total_elements != self.num_elements(new_shape):
      raise ValueError("Total elements in new shape must match the number of elements in the original tensor")

    reshaped_data = self.reshape_recursive(flat_data, new_shape)
    return tensor(reshaped_data, requires_grad=self.requires_grad, dtype=self.dtype)

  def reshape_recursive(self, data, shape):
    if len(shape) == 1:
      return data[:shape[0]]
    step = len(data) // shape[0]
    return [self.reshape_recursive(data[i*step:(i+1)*step], shape[1:]) for i in range(shape[0])]

  def num_elements(self, shape):
    result = 1
    for dim in shape:
      result *= dim
    return result

# Example usage
a = tensor([[1, 2], [3, 4], [5, 6]])
b = a.view(2, 3)

print(b.data)  # Output: [[1, 2, 3], [4, 5, 6]]

class tensor:
  def __init__(self, data, requires_grad=False, dtype=float):
    self.data = data
    self.shape = self.get_shape(data)
    self.requires_grad = requires_grad
    self.dtype = dtype
  
  def get_shape(self, data):
    if isinstance(data, list):
      return [len(data)] + self.get_shape(data[0]) if data else []
    return []

  def flatten(self):
    # Flatten the data for reshaping
    if isinstance(self.data, list):
      return [item for sublist in self.data for item in sublist]
    return [self.data]

  def view(self, *new_shape):
    self.contiguous()  # Ensure the tensor is contiguous before reshaping
    flat_data = self.flatten()
    total_elements = len(flat_data)

    if total_elements != self.num_elements(new_shape):
      raise ValueError("Total elements in new shape must match the number of elements in the original tensor")

    reshaped_data = self.reshape_recursive(flat_data, new_shape)
    return tensor(reshaped_data, requires_grad=self.requires_grad, dtype=self.dtype)

  def reshape_recursive(self, data, shape):
    if len(shape) == 1:
      return data[:shape[0]]
    step = len(data) // shape[0]
    return [self.reshape_recursive(data[i*step:(i+1)*step], shape[1:]) for i in range(shape[0])]

  def num_elements(self, shape):
    result = 1
    for dim in shape:
      result *= dim
    return result

  def contiguous(self):
    # If the tensor is already flat or non-nested, return it as is
    if isinstance(self.data, list):
      flattened_data = self.flatten()
      reshaped_data = self.reshape_recursive(flattened_data, self.shape)
      self.data = reshaped_data
    return self

# Example usage
a = tensor([[1, 2], [3, 4], [5, 6]])
b = a.contiguous()  # Ensure it's stored contiguously in memory

print(b.data)  # Output will be the same as input data since it's already contiguous
