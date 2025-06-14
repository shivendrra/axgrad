# import axgrad

# t1 = axgrad.tensor([[1, 2, 3]], requires_grad=True)
# t2 = axgrad.tensor([[4, 5, 6]], requires_grad=True)

# stacked = axgrad.concat([t1, t2], axis=0)
# d = stacked.tanh()
# out = d.sum()
# print(out)
# print(stacked)

# out.backward()
# print(out)
# print(d)
# print(stacked)
# print(t1)  # Should show gradients for t1
# print(t2)  # Should show gradients for t2

# import axgrad
# from axgrad.utils import RNG, yingyang_dataset

# random = RNG(37)
# train_split, val_split, test_split = yingyang_dataset(random, n=20)
# train_split, val_split, test_split = axgrad.tensor(train_split), axgrad.tensor(val_split), axgrad.tensor(test_split)
# print(train_split, val_split, test_split)

from axgrad import tensor

a = tensor([[2, 4, 5, -4], [-3, 0, 9, -1]], requires_grad=True, dtype="float32")
b = tensor([[1, 0, -2, 0], [-1, 10, -2, 4]], requires_grad=True, dtype="float32")
print(a)
print(b)

c = a + b
d = c.tanh()
# e = d.silu()
f = d ** 2
g = f.sigmoid()
h = g.sum()

print("a:")
print(a)
print("\nb:")
print(b)
print("\nc:")
print(c)
print("\nd:")
print(d)
# print("\ne:")
# print(e)
print("\nf:")
print(f)
print("\ng:")
print(g)
print("\nh:")
print(h)