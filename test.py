# from axon import tensor, nn

# class MLP(nn.Module):
#   def __init__(self):
#     super().__init__()
#     self.fc1 = nn.Linear(3, 5)
#     self.relu = nn.ReLU()
#     self.fc2 = nn.Linear(5, 1)

#   def forward(self, x):
#     x = self.fc1(x)
#     x = self.relu(x)
#     x = self.fc2(x)
#     return x

# a = [[2, 3, 5], [6, 4, 9]]
# y = tensor([[1], [-1]])

# model = MLP()
# # print("total params:", model.n_param())

# out = model(a)
# def mse_loss( trg, prd):
#   diff = trg - prd
#   sq = diff ** 2
#   loss = sq / len(prd.data)
#   return loss

# loss = mse_loss(y, out)
# loss.backward()

# print(loss)

import axgrad
from axgrad import tensor

a = [[1, -4],
     [-2, 7]]

b = [[5, 1],
     [-9, -5]]

a, b = tensor(a, requires_grad=True), tensor(b, requires_grad=True)

''' element level '''
print("add: ", a + [2, 0])
print("mul: ", a * [2, 0])
print("sub: ", a - [2, 0])
print("div: ", a / [2, 0])
print("pow: ", a ** 2)

''' more operations '''
print("shape: ", a.shape)
print("transpose", a.T)
print("sum: ", a.sum())
print("matmul: ", axgrad.matmul(a, b))