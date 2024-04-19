from axgrad import nn as nn
from axgrad.optimizer import Loss, Optim
from axgrad.engine import Value
n = nn.MLP(3, [4, 4, 1])

xs = [
  [-2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, -1.0, 1.0],
  [1.0, 1.0, -1.0],
]

ys = [-1.0, 1.0, -1.0, 1.0]

def absolute(val: list):
  return [abs(v) for v in val]

l_arr = []

ypred = [n(x) for x in xs]

# for k in range(10):
#   ypred = [n(x) for x in xs]
#   loss = Value(sum(abs((yout - ygt).data)for ygt, yout in zip(ys, ypred)))
#   l_arr.append(loss.data)
#   n.zero_grad()
#   loss.backward()

#   for p in n.parameters():
#     p.data += -0.05 * p.grad
  
#   # print(k, loss.data)
  
print(ypred)
print(l_arr)

optimizer = Optim.sgd(n.parameters(), learn_rate=0.05)
loss_f = Loss()

for k in range(10):
  ypred = [n(x) for x in xs]
  loss = loss_f.mse_loss(ys, ypred)
  l_arr.append(loss.data)
  n.zero_grad()
  loss.backward()
  optimizer.step()

print(ypred)
print(l_arr)

import matplotlib.pyplot as plt

plt.plot(range(10), l_arr)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.show()