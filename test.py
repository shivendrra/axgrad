# from axgrad.modules.matrices import ones
# from axgrad.arrays import array as arr

# zero_list = ones((1, 3), dtype=float)
# one_list = ones((1, 3), dtype=float)

# zero_list = arr(zero_list)
# one_list = arr(one_list)

import axgrad.nn as nn

linear_layer = nn.Linear(4, 5)
input_data = [1.0, 2.0, 3.0, 8.0]
output = linear_layer(input_data)

print(output)