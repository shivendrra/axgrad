from .arrays import tensor
from .helpers.statics import zeros

def convolution_2d(image:list, kernel:list) -> list:
  img_h, img_w = len(image), len(image[0])
  ker_h, ker_w = len(kernel), len(kernel[0])

  output_height = img_h - ker_h + 1
  output_width = img_w - ker_w + 1

  output = [[0] * output_width for _ in range(output_height)]

  for i in range(output_height):
    for j in range(output_width):
      for k in range(ker_h):
        for l in range(ker_w):
          output[i][j] += image[i+k][j+l] * kernel[k][l]
  return output

def matmul(x:tensor, y:tensor) -> tensor:
  x = x if isinstance(x, tensor) else tensor(x)
  y = y if isinstance(y, tensor) else tensor(y)
  if len(x.data[0]) != len(y.data):
    raise ValueError(f"Matrices have incompatible dimensions for multiplication. {x.shape} != {y.shape}")

  out = zeros((len(x.data), len(y.data[0])))
  y_t = y.transpose().data
  for i in range(len(x.data)):
    for j in range(len(y_t)):
      out[i][j] = sum(x.data[i][k] * y_t[j][k] for k in range(len(y.data)))
  return tensor(out)

def stack(tuples: tuple, dim:int =0) -> tensor:
  pass

def cat(tuples:tuple, dim:int =0) -> tensor:
  pass