""" matrix operations in one file"""

from .shape import get_shape, check_arr

def matmul_2d(arr1, arr2):
  if check_arr(arr1, arr2) is True:
    out = [[sum(arr1[i][k] * arr2[k][j] for k in range(len(arr2))) for j in range(len(arr2[0]))] for i in range(len(arr1))]
    return out
  else:
    raise ArithmeticError(f"Shape are not matching: {len(arr1[0])} should be equal to {len(arr2[1])}")

def element_wise_operate(arr1, arr2, _op):
  if get_shape(arr1) == get_shape(arr2):
    out = [[0 for _ in range(len(arr2[0]))] for _ in range(len(arr1))]
    
    if _op == 'addition':
      for i in range(len(arr1)):
        for j in range(len(arr1[i])):
          out[i][j] += arr1[i][j] + arr2[i][j]
    elif _op == 'subtraction':
      for i in range(len(arr1)):
        for j in range(len(arr1[i])):
          out[i][j] += arr1[i][j] + (-arr2[i][j])
    elif _op == 'multiplication':
      for i in range(len(arr1)):
        for j in range(len(arr1[i])):
          out[i][j] += arr1[i][j] * arr2[i][j]
    elif _op == 'division':
      for i in range(len(arr1)):
        for j in range(len(arr1[i])):
          out[i][j] += arr1[i][j] / arr2[i][j]
    
    del arr1, arr2
    return out
  
  else:
    raise ArithmeticError(f"both matrix should be of same shape for element level {_op}")