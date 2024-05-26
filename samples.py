import axon
from axon import tensor, nn

# a = tensor([[-1, 2, -3, 4], [5, -6, 7, -8]])
# b = tensor([[1, 4, 6, 6], [4, 7, -2, -3]])

# a = tensor([[1, 2], [3, 4]])
# b = tensor([[5, 6], [7, 8]])

# c = axon.matmul(a, b)
# c.backward()
# print(a.grad)

class MyModel(nn.Module):
  def __init__(self):
    super(MyModel, self).__init__()
    self.fc1 = nn.Linear(10, 5)
    self.fc2 = nn.Linear(5, 2)

  def forward(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    return x

model = MyModel()
print(model.parameters())