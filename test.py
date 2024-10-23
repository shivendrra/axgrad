import axgrad
import axgrad.nn as nn

# t1 = axgrad.tensor([1, 2, 3], requires_grad=True)
# t2 = axgrad.tensor([4, 5, 6], requires_grad=True)

# stacked = axgrad.concat([t1, t2], axis=0)
# d = stacked.tanh()
# out = d.sum()
# print(out)
# print(stacked)

# out.backward()
# print(out.grad)
# print(d.grad)
# print(stacked.grad)
# print(t1.grad)  # Should show gradients for t1
# print(t2.grad)  # Should show gradients for t2

input_tensor = axgrad.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2, stride=1, padding=0)
output = conv(input_tensor)
print(output)