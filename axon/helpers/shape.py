""" tensor's basic shape operations, all in one file """

import numpy as np

def get_shape(arr):
  # returns the shape of tensor
  if isinstance(arr, list):
    return (len(arr),) + get_shape(arr[0])
  else:
    return ()

def check_arr(arr1, arr2):
  if get_shape(arr1)[1] == get_shape(arr2)[0] and len(get_shape(arr2)) == 2:
    return True

def view_arr(arr):
  raise NotImplementedError("view function is empty")

def broadcast_shapes(shape1, shape2):
  """
    broadcasted shape for two input shapes according to numpy's broadcasting rules
      - reverse the shapes to align dimensions from the least significant (rightmost) to the most significant (leftmost)
      - iterate over pairs of dimensions from both shapes, if:
        - dims are equal, they remain unchanged
        - one dim is 1, expanded to match the other dim
        - dims are different & none of them is 1, raise ValueError
      - add remaining dim from longer shape
      - reverse result to restore original order

    args:
      shape1 (tuple): shape of the first array
      shape2 (tuple): shape of the second array

    returns:
      tuple: The broadcasted shape
  """
  result_shape = []
  for dim1, dim2 in zip(reversed(shape1), reversed(shape2)):
    if dim1 == 1 or dim2 == 1 or dim1 == dim2:
      result_shape.append(max(dim1, dim2))
    else:
      raise ValueError(f"Shapes {shape1} and {shape2} are not compatible for broadcasting")
  result_shape.extend(reversed(shape1[len(shape2):]))
  result_shape.extend(reversed(shape2[len(shape1):]))
  return tuple(reversed(result_shape))

def broadcast_array(array, target_shape):
  """
    broadcast an array to a target shape from scratch
      - if current shape matches target shape, return array as is
      - else, recursively expand dims of  array to match target shape
      - nested loops and recursive calls to replicate elements along expanded dims to achieve target shape

    args:
      array (list): input array to be broadcasted
      target_shape (tuple): target shape to broadcast to

    returns:
      list: broadcasted array
  """
  def expand_dims(array, current_shape, target_shape):
    """
    expand dimensions of an array to match the target shape
    args:
      array (list): input array to be expanded
      current_shape (tuple): current shape of the array
      target_shape (tuple): target shape to expand to
    returns:
      list: expanded array
    """
    if len(current_shape) < len(target_shape):
      array = [array]
      current_shape = (1,) + current_shape
    for i, (curr_dim, target_dim) in enumerate(zip(current_shape, target_shape)):
      if curr_dim == 1:
        array = [array] * target_dim
      else:
        array = [expand_dims(subarray, current_shape[1:], target_shape[1:]) for subarray in array]
    return array

  current_shape = get_shape(array)
  if current_shape == target_shape:
    return array
  return expand_dims(array, current_shape, target_shape)