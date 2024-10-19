import axgrad

# Example usage
t1 = axgrad.tensor([1, 2, 3], requires_grad=True)
t2 = axgrad.tensor([4, 5, 6], requires_grad=True)

stacked = axgrad.stack([t1, t2], axis=0)
# d = stacked.tanh()
# out = d.sum()
# print(out)  # Should print the stacked tensor

# out.backward()
# print(out.grad)
# print(d.grad)
# print(t1.grad)  # Should show gradients for t1
# print(t2.grad)  # Should show gradients for t2