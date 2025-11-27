import axgrad.nn as nn
from axgrad.utils import randn, randint
from axgrad.nn import functional as F
import os


class MLP(nn.Module):
  def __init__(self, _in, _hid, _out, bias=False) -> None:
    super().__init__()
    self.embedd = nn.Embedding(100, _in)
    self.layer1 = nn.Linear(_in, _hid, bias)
    self.gelu = nn.GELU()
    self.layer2 = nn.Linear(_hid, _out, bias)

  def forward(self, x):
    out = self.embedd(x)
    out = self.layer1(out)
    out = self.gelu(out)
    out = self.layer2(out)
    return out

# Create model and optimizer
model = MLP(20, 10, 10)
optimizer = nn.SGD(parameters=model.parameters(), lr=1e-5)
x = randint(0, 100, 10)
y = randn(10, 10)

# Check if saved model exists
model_path = "models/mlp_checkpoint.pkl"
start_epoch = 1
losses = []
steps = []

if os.path.exists(model_path):
  print("Found existing model, loading...")
  model.load(model_path)
  # Optionally, you could also save/load optimizer state and training progress
  print("Model loaded successfully!")
  start_epoch = 1  # You could save/load the actual epoch number too
else:
  print("No existing model found, starting fresh training...")

# First training phase
print("\n=== PHASE 1: Initial Training ===")
epoch = 5000
end_epoch = start_epoch + epoch - 1

for n in range(start_epoch, end_epoch + 1):
  out = model.forward(x)
  loss = F.mae(out, y)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if n % 500 == 0:
    losses.append(loss.tolist())
    steps.append(n)
    print(f"{n}th step, loss: {loss.tolist():.6f}")

print(f"\nTotal parameters: {model.n_params()}")

# Save the model
print(f"\nSaving model to {model_path}...")
model.save(model_path)

# Simulate loading the model (like restarting the program)
print("\n=== PHASE 2: Loading and Continuing Training ===")
print("Simulating program restart...")

# Create a new model instance (as if starting fresh)
model_reloaded = MLP(20, 10, 10)
print(f"New model initialized with {model_reloaded.n_params()} parameters")

# Load the saved weights
print("Loading saved model...")
model_reloaded.load(model_path)

# Create new optimizer for the reloaded model
optimizer_reloaded = nn.SGD(parameters=model_reloaded.parameters(), lr=1e-5)

# Continue training with the reloaded model
print("Continuing training with reloaded model...")
continue_epoch = 5000
start_step = end_epoch + 1
final_epoch = start_step + continue_epoch - 1

for n in range(start_step, final_epoch + 1):
  out = model_reloaded.forward(x)
  loss = F.mae(out, y)

  optimizer_reloaded.zero_grad()
  loss.backward()
  optimizer_reloaded.step()

  if n % 500 == 0:
    losses.append(loss.tolist())
    steps.append(n)
    print(f"{n}th step, loss: {loss.tolist():.6f}")

print(f"\nFinal model has {model_reloaded.n_params()} parameters")
print("Training completed!")

# Save final model
final_model_path = "models/mlp_final.pkl"
print(f"\nSaving final model to {final_model_path}...")
model_reloaded.save(final_model_path)

print("\n=== Training Summary ===")
print(f"Total training steps: {final_epoch}")
print(f"Initial loss: {losses[0]:.6f}")
print(f"Final loss: {losses[-1]:.6f}")
print(f"Models saved to: {model_path} and {final_model_path}")

import matplotlib.pyplot as plt

y_list = y.flatten().tolist()
out_list = out.flatten().tolist()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(steps, losses, marker="o", linestyle="-", color="b")
plt.title("Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.xticks(steps)

plt.subplot(1, 2, 2)
plt.scatter(
  range(y.size), y_list, color="b", label="Actual Values"
)  # Actual values in blue
plt.scatter(
  range(out.size), out_list, color="r", label="Predicted Values"
)  # Predicted values in red
plt.title("Predictions vs Targets")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
