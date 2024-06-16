from axon import tensor, nn

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(3, 5)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(5, 1)

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x

a = [[2, 3, 5], [6, 4, 9]]
y = tensor([[1], [-1]])

model = MLP()
# print("total params:", model.n_param())

out = model(a)
print("out.prev: ", out.prev)
def mse_loss( trg, prd):
  diff = trg - prd
  sq = diff ** 2
  loss = sq / len(prd.data)
  return loss

loss = mse_loss(y, out)
loss.backward()

print(loss.leaf)
# for param in model.parameters():
#   print(param.grad)