from ..engine import Value

def get_shape(arr):
  if isinstance(arr, list):
    return (len(arr), ) + get_shape(arr[0])
  else:
    return ()

def matmul(arr1, arr2):
  xs, ys = get_shape(arr1), get_shape(arr2)
  if xs[1] == ys[0]:
    out = [[sum(arr1[i][k] * arr2[k][j] for k in range(len(arr2))) for j in range(len(arr2[0]))] for i in range(len(arr1))]
    return out
  else:
    raise ArithmeticError(f"tensor shape error! : {xs[0]} != {ys[1]}")

def transpose(arr):
  R, C = get_shape(arr)
  return [[arr[i][j] for i in range(R)] for j in range(C)]