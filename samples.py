# import torch
# x = torch.tensor([[1, 4, 5], [1, 4, 6]])
# print(x)
# print(x.size())

# y = torch.cat((x, x), dim=0)
# print(y)
# print(y.size())

import axon

x = axon.tensor([[1, 4, 5], [1, 4, 6]])

def cat(array: tuple, dim: int=0):
  if not array:
    raise ValueError("Need atleast one array to stack")
  
  # shape checking
  base_shape = array[0].shape # shape of first array for target array
  for arr in array:
    if arr.shape != base_shape:
      raise ValueError("All inputs must be of same shape & size!")
  new_shape = list(base_shape[:])
  print(new_shape)
  new_shape.insert(dim, len(array))
  print(new_shape)
  pass

cat((x, x))
# y = axon.stack((x, x), dim=1)
# print(y)