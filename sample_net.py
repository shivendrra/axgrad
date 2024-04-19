from axon.axgrad import MLP, Optim, Loss

n = MLP(3, [4, 4, 1])

xs = [
  [-2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, -1.0, 1.0],
  [1.0, 1.0, -1.0],
]

ys = [-1.0, 1.0, -1.0, 1.0]

def absolute(val: list):
  return [abs(v) for v in val]

l_arr = [] # for visualization

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
print(loss.data)
import matplotlib.pyplot as plt

plt.plot(range(10), l_arr)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.show()