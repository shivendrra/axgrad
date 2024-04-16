from axgrad import nn_mods as nn

n = nn.MLP(3, [4, 4, 1])

xs = [
  [-2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, -1.0, 1.0],
  [1.0, 1.0, -1.0],
]

ys = [-1.0, 1.0, -1.0, 1.0]

for k in range(10):
  ypred = [n(x) for x in xs]
  loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
  n.zero_grad()
  loss.backward()

  for p in n.parameters():
    p.data += -0.05 * p.grad
  
  print(k, loss.data)