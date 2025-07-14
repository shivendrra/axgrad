# import axgrad.nn as nn
# from axgrad.utils import randn

# x = randn(10, 20)
# y = randn(10, 10)
# linear = nn.Linear(20, 10)
# yp = linear(x)
# loss = ((y - yp) ** 2 / x.size).sum()
# loss.backward()

# print(x.grad)
# print(y.grad)
# print(linear.weight.grad)

import axgrad as ax

a = ax.randn(2, 5)

print("Original tensor:")
print(a)
print(f"a[0][1] = {a[0][1]}")
print(f"a[1][1] = {a[1][1]}")

print("\nAssigning values...")
a[0][1] = 10
a[1][1] = 20

print("\nAfter assignment:")
print(a)
print(f"a[0][1] = {a[0][1]}")
print(f"a[1][1] = {a[1][1]}")

print("\nTesting iteration:")
for i, row in enumerate(a):
  print(f"Row {i}: {row}")
  print("row elements:")
  for j in row:
    print("element: ", j)

print("\nTesting tuple indexing:")
print(f"a[0, 1] = {a[0, 1]}")
print(f"a[1, 1] = {a[1, 1]}")

a[0, 2] = 99
print(f"After a[0, 2] = 99: {a}")