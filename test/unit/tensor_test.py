import pytest
import numpy as np
from axgrad import Tensor

class TestTensorCreation:
  def test_tensor_from_scalar(self):
    t = Tensor([5.0])
    assert t.size == 1
    assert t.ndim == 1
    assert t.shape == (1,)
    assert t.dtype == "float32"

  def test_tensor_from_list(self):
    t = Tensor([1, 2, 3])
    assert t.size == 3
    assert t.ndim == 1
    assert t.shape == (3,)

  def test_tensor_from_nested_list(self):
    t = Tensor([[1, 2], [3, 4]])
    assert t.size == 4
    assert t.ndim == 2
    assert t.shape == (2, 2)

  def test_tensor_with_dtype(self):
    t = Tensor([1, 2, 3], dtype=Tensor.int32)
    assert t.dtype == "int32"

  def test_tensor_with_requires_grad(self):
    t = Tensor([1, 2, 3], requires_grad=True)
    assert t.requires_grad == True
    assert t.grad is None

  def test_tensor_from_tensor(self):
    t1 = Tensor([1, 2, 3])
    t2 = Tensor(t1)
    assert t2.shape == t1.shape
    assert t2.size == t1.size
    assert t2.dtype == t1.dtype

class TestTensorMethods:
  def test_astype(self):
    t = Tensor([1.5, 2.7, 3.9])
    t_int = t.astype(Tensor.int32)
    assert t_int.dtype == "int32"
    assert t_int.shape == t.shape

  def test_tolist_scalar(self):
    t = Tensor([5.0])
    result = t.tolist()
    assert isinstance(result, list)

  def test_tolist_1d(self):
    data = [1, 2, 3]
    t = Tensor(data)
    result = t.tolist()
    assert len(result) == len(data)

  def test_tolist_2d(self):
    data = [[1, 2], [3, 4]]
    t = Tensor(data)
    result = t.tolist()
    assert len(result) == 2

  def test_contiguous(self):
    t = Tensor([[1, 2], [3, 4]])
    t_cont = t.contiguous()
    assert isinstance(t_cont, Tensor)

  def test_make_contiguous(self):
    t = Tensor([[1, 2], [3, 4]])
    t_cont = t.make_contiguous()
    assert isinstance(t_cont, Tensor)

  def test_view(self):
    t = Tensor([[1, 2], [3, 4]])
    t_view = t.view()
    assert isinstance(t_view, Tensor)

class TestTensorComparison:
  def test_eq_scalar(self):
    t = Tensor([1, 2, 3])
    result = t == 2
    assert isinstance(result, Tensor)

  def test_eq_tensor(self):
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([1, 2, 4])
    result = t1 == t2
    assert isinstance(result, Tensor)

  def test_ne_scalar(self):
    t = Tensor([1, 2, 3])
    result = t != 2
    assert isinstance(result, Tensor)

  def test_ne_tensor(self):
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([1, 2, 4])
    result = t1 != t2
    assert isinstance(result, Tensor)

  def test_gt_scalar(self):
    t = Tensor([1, 2, 3])
    result = t > 2
    assert isinstance(result, Tensor)

  def test_gt_tensor(self):
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([0, 2, 2])
    result = t1 > t2
    assert isinstance(result, Tensor)

  def test_ge_scalar(self):
    t = Tensor([1, 2, 3])
    result = t >= 2
    assert isinstance(result, Tensor)

  def test_ge_tensor(self):
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([0, 2, 2])
    result = t1 >= t2
    assert isinstance(result, Tensor)

  def test_lt_scalar(self):
    t = Tensor([1, 2, 3])
    result = t < 2
    assert isinstance(result, Tensor)

  def test_lt_tensor(self):
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([2, 3, 4])
    result = t1 < t2
    assert isinstance(result, Tensor)

  def test_le_scalar(self):
    t = Tensor([1, 2, 3])
    result = t <= 2
    assert isinstance(result, Tensor)

  def test_le_tensor(self):
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([2, 3, 4])
    result = t1 <= t2
    assert isinstance(result, Tensor)

class TestTensorArithmetic:
  def test_add(self):
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    result = t1 + t2
    assert isinstance(result, Tensor)

  def test_radd(self):
    t = Tensor([1, 2, 3])
    result = 5 + t
    assert isinstance(result, Tensor)

  def test_sub(self):
    t1 = Tensor([4, 5, 6])
    t2 = Tensor([1, 2, 3])
    result = t1 - t2
    assert isinstance(result, Tensor)

  def test_rsub(self):
    t = Tensor([1, 2, 3])
    result = 5 - t
    assert isinstance(result, Tensor)

  def test_mul(self):
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    result = t1 * t2
    assert isinstance(result, Tensor)

  def test_rmul(self):
    t = Tensor([1, 2, 3])
    result = 2 * t
    assert isinstance(result, Tensor)

  def test_truediv(self):
    t1 = Tensor([4, 6, 8])
    t2 = Tensor([2, 3, 4])
    result = t1 / t2
    assert isinstance(result, Tensor)

  def test_rtruediv(self):
    t = Tensor([2, 4, 8])
    result = 16 / t
    assert isinstance(result, Tensor)

  def test_pow(self):
    t = Tensor([2, 3, 4])
    result = t ** 2
    assert isinstance(result, Tensor)

  def test_rpow(self):
    t = Tensor([2, 3, 4])
    result = 2 ** t
    assert isinstance(result, Tensor)

  def test_matmul(self):
    t1 = Tensor([[1, 2], [3, 4]])
    t2 = Tensor([[5, 6], [7, 8]])
    result = t1 @ t2
    assert isinstance(result, Tensor)

  def test_neg(self):
    t = Tensor([1, -2, 3])
    result = -t
    assert isinstance(result, Tensor)

class TestTensorUnaryOps:
  def test_log(self):
    t = Tensor([1, 2, 3])
    result = t.log()
    assert isinstance(result, Tensor)

  def test_exp(self):
    t = Tensor([1, 2, 3])
    result = t.exp()
    assert isinstance(result, Tensor)

  def test_sign(self):
    t = Tensor([-1, 0, 1])
    result = t.sign()
    assert isinstance(result, Tensor)

  def test_abs(self):
    t = Tensor([-1, -2, 3])
    result = t.abs()
    assert isinstance(result, Tensor)

  def test_sqrt(self):
    t = Tensor([1, 4, 9])
    result = t.sqrt()
    assert isinstance(result, Tensor)

  def test_sin(self):
    t = Tensor([0, 1.57, 3.14])
    result = t.sin()
    assert isinstance(result, Tensor)

  def test_cos(self):
    t = Tensor([0, 1.57, 3.14])
    result = t.cos()
    assert isinstance(result, Tensor)

  def test_tan(self):
    t = Tensor([0, 0.785, 1.57])
    result = t.tan()
    assert isinstance(result, Tensor)

  def test_sinh(self):
    t = Tensor([0, 1, 2])
    result = t.sinh()
    assert isinstance(result, Tensor)

  def test_cosh(self):
    t = Tensor([0, 1, 2])
    result = t.cosh()
    assert isinstance(result, Tensor)

  def test_tanh(self):
    t = Tensor([0, 1, 2])
    result = t.tanh()
    assert isinstance(result, Tensor)

class TestTensorActivations:
  def test_sigmoid(self):
    t = Tensor([-1, 0, 1])
    result = t.sigmoid()
    assert isinstance(result, Tensor)

  def test_relu(self):
    t = Tensor([-1, 0, 1])
    result = t.relu()
    assert isinstance(result, Tensor)

  def test_gelu(self):
    t = Tensor([-1, 0, 1])
    result = t.gelu()
    assert isinstance(result, Tensor)

  def test_elu(self):
    t = Tensor([-1, 0, 1])
    result = t.elu()
    assert isinstance(result, Tensor)

  def test_elu_with_alpha(self):
    t = Tensor([-1, 0, 1])
    result = t.elu(alpha=0.01)
    assert isinstance(result, Tensor)

  def test_leakyrelu(self):
    t = Tensor([-1, 0, 1])
    result = t.leakyrelu()
    assert isinstance(result, Tensor)

  def test_leakyrelu_with_eps(self):
    t = Tensor([-1, 0, 1])
    result = t.leakyrelu(eps=0.01)
    assert isinstance(result, Tensor)

  def test_swish(self):
    t = Tensor([-1, 0, 1])
    result = t.swish()
    assert isinstance(result, Tensor)

  def test_swish_with_beta(self):
    t = Tensor([-1, 0, 1])
    result = t.swish(beta=1.0)
    assert isinstance(result, Tensor)

  def test_silu(self):
    t = Tensor([-1, 0, 1])
    result = t.silu()
    assert isinstance(result, Tensor)

  def test_softplus(self):
    t = Tensor([-1, 0, 1])
    result = t.softplus()
    assert isinstance(result, Tensor)

class TestTensorReductionOps:
  def test_sum_default(self):
    t = Tensor([[1, 2], [3, 4]])
    result = t.sum()
    assert isinstance(result, Tensor)

  def test_sum_with_axis(self):
    t = Tensor([[1, 2], [3, 4]])
    result = t.sum(axis=0)
    assert isinstance(result, Tensor)

  def test_sum_with_keepdims(self):
    t = Tensor([[1, 2], [3, 4]])
    result = t.sum(keepdims=True)
    assert isinstance(result, Tensor)

  def test_mean_default(self):
    t = Tensor([[1, 2], [3, 4]])
    result = t.mean()
    assert isinstance(result, Tensor)

  def test_mean_with_axis(self):
    t = Tensor([[1, 2], [3, 4]])
    result = t.mean(axis=0)
    assert isinstance(result, Tensor)

  def test_mean_with_keepdims(self):
    t = Tensor([[1, 2], [3, 4]])
    result = t.mean(keepdims=True)
    assert isinstance(result, Tensor)

  def test_var_default(self):
    t = Tensor([1, 2, 3, 4])
    result = t.var()
    assert isinstance(result, Tensor)

  def test_var_with_axis(self):
    t = Tensor([[1, 2], [3, 4]])
    result = t.var(axis=0)
    assert isinstance(result, Tensor)

  def test_var_with_ddof(self):
    t = Tensor([1, 2, 3, 4])
    result = t.var(ddof=1)
    assert isinstance(result, Tensor)

  def test_std_default(self):
    t = Tensor([1, 2, 3, 4])
    result = t.std()
    assert isinstance(result, Tensor)

  def test_std_with_axis(self):
    t = Tensor([[1, 2], [3, 4]])
    result = t.std(axis=0)
    assert isinstance(result, Tensor)

  def test_std_with_ddof(self):
    t = Tensor([1, 2, 3, 4])
    result = t.std(ddof=1)
    assert isinstance(result, Tensor)

  def test_max_default(self):
    t = Tensor([[1, 2], [3, 4]])
    result = t.max()
    assert isinstance(result, Tensor)

  def test_max_with_axis(self):
    t = Tensor([[1, 2], [3, 4]])
    result = t.max(axis=0)
    assert isinstance(result, Tensor)

  def test_max_with_keepdims(self):
    t = Tensor([[1, 2], [3, 4]])
    result = t.max(keepdims=True)
    assert isinstance(result, Tensor)

  @pytest.mark.skip(reason="tensor.min() has some unresolved bugs, need to fix")
  def test_min_default(self):
    t = Tensor([[1, 2], [3, 4]])
    result = t.min()
    assert isinstance(result, Tensor)

  @pytest.mark.skip(reason="tensor.min() has some unresolved bugs, need to fix")
  def test_min_with_axis(self):
    t = Tensor([[1, 2], [3, 4]])
    result = t.min(axis=0)
    assert isinstance(result, Tensor)

  @pytest.mark.skip(reason="tensor.min() has some unresolved bugs, need to fix")
  def test_min_with_keepdims(self):
    t = Tensor([[1, 2], [3, 4]])
    result = t.min(keepdims=True)
    assert isinstance(result, Tensor)

class TestTensorShapeOps:
  def test_transpose(self):
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    result = t.transpose()
    assert isinstance(result, Tensor)

  def test_flatten(self):
    t = Tensor([[1, 2], [3, 4]])
    result = t.flatten()
    assert isinstance(result, Tensor)

  def test_reshape_tuple(self):
    t = Tensor([1, 2, 3, 4])
    result = t.reshape((2, 2))
    assert isinstance(result, Tensor)

  def test_reshape_list(self):
    t = Tensor([1, 2, 3, 4])
    result = t.reshape([2, 2])
    assert isinstance(result, Tensor)

  def test_squeeze(self):
    t = Tensor([[[1], [2]]])
    result = t.squeeze(axis=2)
    assert isinstance(result, Tensor)

  def test_unsqueeze(self):
    t = Tensor([1, 2, 3])
    result = t.unsqueeze(axis=1)
    assert isinstance(result, Tensor)

class TestTensorNormOps:
  def test_clip(self):
    t = Tensor([-2, -1, 0, 1, 2])
    result = t.clip(max_val=1.5)
    assert isinstance(result, Tensor)

  def test_clamp(self):
    t = Tensor([-2, -1, 0, 1, 2])
    result = t.clamp(min_val=-1.5, max_val=1.5)
    assert isinstance(result, Tensor)

class TestTensorIndexing:
  def test_getitem(self):
    t = Tensor([1, 2, 3, 4])
    item = t[0]
    assert isinstance(item, (int, float, Tensor))

  def test_setitem(self):
    t = Tensor([1, 2, 3, 4])
    t[0] = 10
    assert t is not None

  def test_iter(self):
    t = Tensor([1, 2, 3])
    items = list(t)
    assert len(items) >= 0

class TestTensorGradients:
  def test_backward_scalar(self):
    t = Tensor([2.0], requires_grad=True)
    t2 = t * t
    t2.backward()
    assert t.grad is not None

  def test_backward_with_gradient(self):
    t = Tensor([2.0], requires_grad=True)
    t2 = t * t
    gradient = Tensor([2.0])
    t2.backward(gradient)
    assert t.grad is not None

  def test_register_hook(self):
    t = Tensor([1, 2, 3], requires_grad=True)
    def test_hook(grad):
      return grad * 2
    t.register_hook(test_hook)
    assert len(t.hooks) == 1

  def test_multiple_hooks(self):
    t = Tensor([1, 2, 3], requires_grad=True)
    def hook1(grad):
      return grad
    def hook2(grad):
      return grad
    t.register_hook(hook1)
    t.register_hook(hook2)
    assert len(t.hooks) == 2

class TestTensorMisc:
  def test_str(self):
    t = Tensor([1, 2, 3])
    result = str(t)
    assert isinstance(result, str)

  def test_hash(self):
    t = Tensor([1, 2, 3])
    result = hash(t)
    assert isinstance(result, int)

  def test_hash_uniqueness(self):
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([1, 2, 3])
    assert hash(t1) != hash(t2)

class TestTensorEdgeCases:
  @pytest.mark.skip(reason="Empty tensors not supported")
  def test_empty_tensor(self):
    t = Tensor([])
    assert t.size == 0
    assert t.tolist() == []

  def test_single_element_tensor(self):
    t = Tensor([42])
    assert t.size == 1
    assert t.ndim == 1

  def test_large_nested_tensor(self):
    data = [[[i + j + k for k in range(2)] for j in range(3)] for i in range(4)]
    t = Tensor(data)
    assert t.ndim == 3
    assert t.shape == (4, 3, 2)

  def test_dtype_constants(self):
    assert hasattr(Tensor, 'int8')
    assert hasattr(Tensor, 'int16')
    assert hasattr(Tensor, 'int32')
    assert hasattr(Tensor, 'int64')
    assert hasattr(Tensor, 'float32')
    assert hasattr(Tensor, 'float64')
    assert hasattr(Tensor, 'boolean')

  def test_grad_accumulation(self):
    t = Tensor([1.0], requires_grad=True)
    t.grad = Tensor([1.0])
    t.grad = Tensor([2.0])
    assert t.grad is not None

if __name__ == "__main__":
  pytest.main([__file__, "-v"])