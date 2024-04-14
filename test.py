from axgrad.modules.matrices import ones
from axgrad.arrays import array as arr

zero_list = ones((1, 3, 2, 5), dtype=float)
one_list = ones((1, 3, 2, 5), dtype=float)

zero_list = arr(zero_list)
one_list = arr(one_list)

print(zero_list)