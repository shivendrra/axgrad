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

# import axgrad
# from axgrad.utils import RNG, yingyang_dataset

# random = RNG(37)
# train_split, val_split, test_split = yingyang_dataset(random, n=20)
# train_split, val_split, test_split = axgrad.tensor(train_split), axgrad.tensor(val_split), axgrad.tensor(test_split)
# print(train_split, val_split, test_split)

from axgrad import tensor

a = tensor([[2, 4, 5, -4], [-3, 0, 9, -1]])
b = tensor([[1, 0, -2, 0], [-1, 10, -2, 4]])

c = a + b
d = c.tanh()
e = d.silu()
f = e ** 2
g = f.sigmoid()
h = g.sum()

h.backward()

print("a.grad:")
print(a.grad)
print("\nb.grad:")
print(b.grad)
print("\nc.grad:")
print(c.grad)
print("\nd.grad:")
print(d.grad)
print("\ne.grad:")
print(e.grad)
print("\nf.grad:")
print(f.grad)
print("\ng.grad:")
print(g.grad)
print("\nh.grad:")
print(h.grad)