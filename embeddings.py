import numpy as np
from arrays import zeros, array
import math

class PositionalEmbeddings:
  def __init__(self, dim1: int, dim2: int):
    self.dim1 = dim1
    self.dim2 = dim2
  
  def apply_embd(self, seq):
    # pos_embd = zeros((self.dim1, self.dim2))
    pos_embd = array(seq)
    for pos in range(self.dim1):
      for i in range(0, self.dim2, 2):
        pos = array(pos) 
        exp = i / self.dim2
        pos_embd[pos][i] = math.sin(pos/(1e4**exp))
        pos_embd[pos][i+1] = math.cos(pos/(1e4**exp))
    return pos_embd
  
  def __call__(self, x):
    return self.apply_embd(x)

input_seq = [[1, 3, 5], [2, 5, 6], [3, 5, 6]]

pos_enc = PositionalEmbeddings(10, 15)
res = pos_enc(input_seq)
print(res)