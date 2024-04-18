def get_shape(arr):
  """
    args:
    - arr: array for determining the shape

    returns:
    - tuple with the shape
  """
  return [] if not isinstance(arr, list) else [len(arr)] + get_shape(arr[0])

def _operate(arr1, arr2, op=''):
  """
    staticmethod to carry addition or subtraction for __add__ & __sub__

    args:
      - arr1: first tensor
      - arr2: second tensor
      - _op: '+' or '-'
    
    returns:
      - matrix with performed operation
  """
  if len(arr1) != len(arr2):
    raise ValueError("Arrays must be of same shape & size")
  result = []
  for i in range(len(arr1)):
    result.append(_operate(arr1[i], arr2[i], op=op)) if isinstance(arr1[i], list) and isinstance(arr2[i], list) else result.append(arr1[i] + arr2[i]) if op=='+' else result.append(arr1[i] - arr2[i])
  return result