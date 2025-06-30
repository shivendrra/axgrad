import axgrad
import axgrad.nn as nn
from axgrad.nn.modules import linear as F
import matplotlib.pyplot as plt

class MLP(nn.Module):
  def __init__(self, _in, _hid, _out, bias=False) -> None:
    super().__init__()
    self.layer1 = nn.Linear(_in, _hid, bias)
    self.gelu = nn.GELU()
    self.layer2 = nn.Linear(_hid, _out, bias)
  
  def forward(self, x):
    out = self.layer1(x)
    out = self.gelu(out)
    out = self.layer2(out)
    return out

model = MLP(20, 30, 1)

# Generate random input and target tensors
X = axgrad.tensor(axgrad.randn(shape=(15, 20)), requires_grad=True)  # Input tensor of shape (batch_size=15, features=20)
Y = axgrad.tensor(axgrad.randn(shape=(15, 1)), requires_grad=True)  # Target tensor of shape (batch_size=15, 1)

optimizer = nn.SGD(parameters=model.parameters(), lr=2e-5)
epoch = 6000
losses = []
steps = []

for n in range(1, epoch + 1):
  out = model.forward(X)
  loss = F.mse(out, Y)
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if n % 300 == 0:
    losses.append(loss.data[0])  # Store the loss
    steps.append(n)  # Store the step
    print(f"{n}th step, loss: {loss.data[0]:.6f}")

# Plotting the learning curve
plt.figure(figsize=(12, 5))

# Learning curve
plt.subplot(1, 2, 1)
plt.plot(steps, losses, marker='o', linestyle='-', color='b')
plt.title('Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.xticks(steps)

# Predictions vs Targets
plt.subplot(1, 2, 2)
# Scatter plot for actual values (Y) and predicted values (out)
plt.scatter(range(len(Y.F.data)), Y.F.data, color='b', label='Actual Values')  # Actual values in blue
plt.scatter(range(len(out.F.data)), out.F.data, color='r', label='Predicted Values')  # Predicted values in red
plt.title('Predictions vs Targets')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()