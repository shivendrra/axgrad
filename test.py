# import axgrad

# t1 = axgrad.tensor([[1, 2, 3]], requires_grad=True)
# t2 = axgrad.tensor([[4, 5, 6]], requires_grad=True)

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

import axgrad
from axgrad.utils import RNG, yingyang_dataset

random = RNG(37)
train_split, val_split, test_split = yingyang_dataset(random, n=20)
train_split, val_split, test_split = axgrad.tensor(train_split), axgrad.tensor(val_split), axgrad.tensor(test_split)
print(train_split, val_split, test_split)