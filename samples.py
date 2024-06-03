import axon
from axon import tensor, nn

a = tensor([[-1, 2, -3, 4], [5, -6, 7, -8]])
b = tensor(1.5)
c = a + b
d = c * b
d.backward()
print(d)
print(a.grad)
print(b.grad)
print(c.grad)
print(d.grad)

# b = tensor([[1, 4, 6, 6], [4, 7, -2, -3]])

# a = tensor([[1, 2], [3, 4]])
# b = tensor([[5, 6], [7, 8]])

# c = axon.matmul(a, b)
# c.backward()
# print(a.grad)
# print(a + b)

# class MLP(nn.Module):
#   def __init__(self):
#     super().__init__()
#     self.fc1 = nn.Linear(4, 5)
#     self.relu = nn.ReLU()
#     self.fc2 = nn.Linear(5, 1)

#   def forward(self, x):
#     x = self.fc1(x)
#     x = self.relu(x)
#     x = self.fc2(x)
#     return x

# model = MLP()
# print(model)
# print("total params:", model.n_param())
# out = model(a)
# print(out)