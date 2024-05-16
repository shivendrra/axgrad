# import numpy as np

# # Define matrices A, B, and D
# A = np.random.randn(3, 4)  # Shape (3, 4)
# B = np.random.randn(4, 5)  # Shape (4, 5)
# D = np.random.randn(3, 5)  # Shape (3, 5)

# # Function for matrix multiplication
# def matmul(A, B):
#     m, n = A.shape
#     n, p = B.shape
#     C = np.zeros((m, p))
#     for i in range(m):
#         for j in range(p):
#             for k in range(n):
#                 C[i, j] += A[i, k] * B[k, j]
#     return C

# # Function for matrix addition
# def matadd(C, D):
#     m, p = C.shape
#     E = np.zeros((m, p))
#     for i in range(m):
#         for j in range(p):
#             E[i, j] = C[i, j] + D[i, j]
#     return E

# # Perform matrix multiplication
# C = matmul(A, B)

# # Add the result with another matrix D
# E = matadd(C, D)

# # Define the loss as the sum of all elements in E
# loss = np.sum(E)

# # Compute gradients manually
# dE = np.ones_like(E)  # Derivative of loss w.r.t E is 1 for each element

# # Gradients w.r.t C (same as dE)
# dC = dE

# # Gradients w.r.t A and B
# dA = np.zeros_like(A)
# dB = np.zeros_like(B)

# # Compute gradients using the chain rule
# m, n = A.shape
# n, p = B.shape

# for i in range(m):
#     for k in range(n):
#         for j in range(p):
#             dA[i, k] += dC[i, j] * B[k, j]
#             dB[k, j] += dC[i, j] * A[i, k]

# # Print the gradients
# print("Gradient w.r.t A:\n", dA)
# print("Gradient w.r.t B:\n", dB)

# # Optionally, print intermediate matrices
# print("Matrix A:\n", A)
# print("Matrix B:\n", B)
# print("Matrix C (A @ B):\n", C)
# print("Matrix D:\n", D)
# print("Matrix E (C + D):\n", E)
# print("Loss (sum of E):\n", loss)

from axon import tensor, zeros
import numpy as np

x = [[1, 4, 4], [1, 5, 6], [1, 5, 7]]
y = [[1, 4, 4], [1, 5, 6], [1, 5, 7], [1, 4, 5]]
z = [[[1, 1, 1, 1], [4, 5, 5, 4], [4, 6, 7, 5]], [[1, 1, 1, 1], [4, 5, 5, 4], [4, 6, 7, 5]]]

a, b, c = tensor(x), tensor(y), tensor(z)

x = a * a * a
print(x.sum())

x = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = tensor([[9, 8, -7], [6, -5, 4], [3, -2, -1]])
print(x + y)

x = tensor([[1, 4, 4], [1, 5, 6], [1, 5, 7]])
z = tensor([[1, 1, 1, 1], [4, 5, 5, 4], [4, 6, 7, 5]])

print(tensor.matmul(x, z))