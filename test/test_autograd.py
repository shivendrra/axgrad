import unittest
import numpy as np
import torch
from axgrad import tensor

class TestTensorOperations(unittest.TestCase):
  def setUp(self):
    self.data = [[1.0, 2.0], [3.0, 4.0]]  # 2x2 Matrix
    self.torch_input = torch.tensor(self.data, requires_grad=True, dtype=torch.float32)
    self.axgrad_input = tensor(self.data, requires_grad=True, dtype='float32')
  
  def test_addition(self):
    # PyTorch Addition
    torch_result = self.torch_input + 2
    # axgrad Addition
    ax_result = self.axgrad_input + 2
    np.testing.assert_array_almost_equal(torch_result.detach().numpy(), ax_result.data, decimal=5)
    torch_result.sum().backward()
    ax_result.sum().backward()
    np.testing.assert_array_almost_equal(self.torch_input.grad.numpy(), self.axgrad_input.grad.data, decimal=5)

  def test_multiplication(self):
    # PyTorch Multiplication
    torch_result = self.torch_input * 3
    # axgrad Multiplication
    ax_result = self.axgrad_input * 3
    np.testing.assert_array_almost_equal(torch_result.detach().numpy(), ax_result.data, decimal=5)
    torch_result.sum().backward()
    ax_result.sum().backward()
    np.testing.assert_array_almost_equal(self.torch_input.grad.numpy(), self.axgrad_input.grad.data, decimal=5)

  def test_power(self):
    torch_result = self.torch_input ** 2
    ax_result = self.axgrad_input ** 2
    np.testing.assert_array_almost_equal(torch_result.detach().numpy(), ax_result.data, decimal=5)
    torch_result.sum().backward()
    ax_result.sum().backward()
    np.testing.assert_array_almost_equal(self.torch_input.grad.numpy(), self.axgrad_input.grad.data, decimal=5)

  def test_relu(self):
    relu_layer = torch.nn.ReLU()
    torch_result = relu_layer(self.torch_input)
    from axgrad.nn import ReLU
    ax_relu = ReLU()
    ax_result = ax_relu(self.axgrad_input)
    np.testing.assert_array_almost_equal(torch_result.detach().numpy(), ax_result.data, decimal=5)
    torch_result.sum().backward()
    ax_result.sum().backward()
    np.testing.assert_array_almost_equal(self.torch_input.grad.numpy(), self.axgrad_input.grad.data, decimal=5)

  def test_matrix_multiplication(self):
    weight_data = [[0.5, 1.0], [-1.0, 1.5]]
    torch_weight = torch.tensor(weight_data, requires_grad=True, dtype=torch.float32)
    ax_weight = tensor(weight_data, requires_grad=True, dtype='float32')

    # PyTorch Matrix Multiplication
    torch_result = torch.matmul(self.torch_input, torch_weight)
    # axgrad Matrix Multiplication
    ax_result = self.axgrad_input @ ax_weight
    np.testing.assert_array_almost_equal(torch_result.detach().numpy(), ax_result.data, decimal=5)

    torch_result.sum().backward()
    ax_result.sum().backward()
    np.testing.assert_array_almost_equal(self.torch_input.grad.numpy(), self.axgrad_input.grad.data, decimal=5)
    np.testing.assert_array_almost_equal(torch_weight.grad.numpy(), ax_weight.grad.data, decimal=5)


if __name__ == '__main__':
  unittest.main()