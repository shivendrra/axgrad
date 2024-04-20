""" tensor's basic shape operations, all in one file """

import numpy as np

def get_shape(arr):
  # returns the shape of tensor
  if isinstance(arr, list):
    return (len(arr),) + get_shape(arr[0])
  else:
    return ()

def broadcast(arr, trg_shape):
  # returns a broadcasted list
  arr = arr if isinstance(arr, np.ndarray) else np.ndarray(arr)
  return np.broadcast_to(arr, trg_shape).tolist()

def check_arr(arr1, arr2):
  if get_shape(arr1)[1] == get_shape(arr2)[0] and len(get_shape(arr2)) == 2:
    return True

def view_arr(arr):
  raise NotImplementedError("view function is empty")