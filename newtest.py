x = [[[1, 5, 6], [6, 7 , 2]], [[1, 5, 6], [6, 7 , 2]], [[1, 5, 6], [6, 7 , 2]], [[1, 5, 6], [6, 7 , 2]]]
y = [[[0,1],[2,3]],[[4,5],[6,7]]]

def get_shape(data):
  if isinstance(data, list):
    return [len(data), ] + get_shape(data[0])
  else:
    return []

def transpose(data):
  return list(map(list, zip(*data)))

def swap_axes(data, dim0, dim1):
  out = []
  def rec_swap(data, axes):
    print(axes)
    print(data)
    if len(axes) == 1:
      print('data: ', data)
      return data
    transpose_data = transpose(data)
    if len(axes) == 2:
      return transpose_data
    else:
      for sub_data in data:
        print(sub_data)
        return rec_swap(sub_data, axes[1:])

  shape = get_shape(data)
  dim0 = dim0 + len(shape) if dim0 < 0 else dim0
  dim1 = dim1 + len(shape) if dim1 < 0 else dim1

  if dim0 == dim1 or not data:
    return data

  axes = list(range(len(shape)))
  # axes[dim0], axes[dim1] = axes[dim1], axes[dim0]

  # out.append(rec_swap(data, axes))
  rec_swap(data, axes)

# def swap_axes(data, dim0, dim1):
#   def recursive_swap(data, axes):
#     if len(axes) == 1:
#       return data
#     transposed_data = transpose(data)

#     if len(axes) == 2:
#       return transposed_data
#     else:
#       return [recursive_swap(sub_data, axes[1:]) for sub_data in transposed_data]

#   shape = get_shape(data)
#   dim0 = dim0 + len(shape) if dim0 < 0 else dim0
#   dim1 = dim1 + len(shape) if dim1 < 0 else dim1

#   if dim0 == dim1 or not data:
#     return data

#   axes = list(range(len(shape)))
#   axes[dim0], axes[dim1] = axes[dim1], axes[dim0]

#   return recursive_swap(data, axes)

print("tensor.shape ", get_shape(x))
print("swapped tensor: ", swap_axes(x, 0, -1))
print("swapped tensor shape: ", get_shape(swap_axes(x, 0, -1)))

import numpy as np
a = np.array(x)

print("np: ", a.shape)
print("np.swapaxes: ", a.swapaxes(0, -1))
print("np.swapaxes.shape: ", a.swapaxes(0, -1).shape)