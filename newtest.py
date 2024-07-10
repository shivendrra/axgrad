x = [[[1, 5, 6], [6, 7 , 2]], [[1, 5, 6], [6, 7 , 2]], [[1, 5, 6], [6, 7 , 2]], [[1, 5, 6], [6, 7 , 2]]]
y = [[[0,1],[2,3]],[[4,5],[6,7]]]

def get_shape(data):
  if isinstance(data, list):
    return [len(data), ] + get_shape(data[0])
  else:
    return []

def transpose(data):
  return list(map(list, zip(*data)))

def swap_axes(data, dim0, dim1, ndim, depth=0):
  if depth == ndim - 2:
    return [list(row) for row in zip(*data)]
  else:
    return [swap_axes(sub_data, dim0, dim1, ndim, depth+1) for sub_data in data]

print(swap_axes(y, 0, 2, len(get_shape(y))))

import numpy as np

a = np.array(x)
print(a.transpose())