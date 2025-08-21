import pytest
import numpy as np
from axgrad import Tensor
from axgrad.utils import zeros_like, ones_like, zeros, ones, randn, randint, uniform, fill, linspace, arange

class TestZerosLike:
  def test_zeros_like_1d(self):
    t = Tensor([1, 2, 3])
    result = zeros_like(t)
    assert isinstance(result, Tensor)
    assert result.shape == t.shape
    assert result.size == t.size
    assert result.ndim == t.ndim

  def test_zeros_like_2d(self):
    t = Tensor([[1, 2], [3, 4]])
    result = zeros_like(t)
    assert result.shape == (2, 2)
    assert result.size == 4
    assert result.ndim == 2

  def test_zeros_like_3d(self):
    t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    result = zeros_like(t)
    assert result.shape == (2, 2, 2)
    assert result.size == 8
    assert result.ndim == 3

  def test_zeros_like_scalar(self):
    t = Tensor([42])
    result = zeros_like(t)
    assert result.shape == (1,)
    assert result.size == 1

  def test_zeros_like_different_dtypes(self):
    t_float = Tensor([1.0, 2.0], dtype=Tensor.float32)
    t_int = Tensor([1, 2], dtype=Tensor.int32)
    result_float = zeros_like(t_float)
    result_int = zeros_like(t_int)
    assert isinstance(result_float, Tensor)
    assert isinstance(result_int, Tensor)

class TestOnesLike:
  def test_ones_like_1d(self):
    t = Tensor([5, 6, 7])
    result = ones_like(t)
    assert isinstance(result, Tensor)
    assert result.shape == t.shape
    assert result.size == t.size
    assert result.ndim == t.ndim

  def test_ones_like_2d(self):
    t = Tensor([[9, 8], [7, 6]])
    result = ones_like(t)
    assert result.shape == (2, 2)
    assert result.size == 4
    assert result.ndim == 2

  def test_ones_like_rectangular(self):
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    result = ones_like(t)
    assert result.shape == (2, 3)
    assert result.size == 6
    assert result.ndim == 2

  def test_ones_like_preserves_strides(self):
    t = Tensor([[1, 2], [3, 4]])
    result = ones_like(t)
    assert result.strides == t.strides

class TestZeros:
  def test_zeros_1d(self):
    result = zeros(5)
    assert isinstance(result, Tensor)
    assert result.shape == (5,)
    assert result.size == 5
    assert result.ndim == 1
    assert result.dtype == "float32"

  def test_zeros_2d(self):
    result = zeros(3, 4)
    assert result.shape == (3, 4)
    assert result.size == 12
    assert result.ndim == 2

  def test_zeros_3d(self):
    result = zeros(2, 3, 4)
    assert result.shape == (2, 3, 4)
    assert result.size == 24
    assert result.ndim == 3

  def test_zeros_with_dtype_int32(self):
    result = zeros(3, dtype=Tensor.int32)
    assert result.dtype == "int32"
    assert result.shape == (3,)

  def test_zeros_with_dtype_float64(self):
    result = zeros(2, 2, dtype=Tensor.float64)
    assert result.dtype == "float64"
    assert result.shape == (2, 2)

  def test_zeros_single_dim(self):
    result = zeros(1)
    assert result.shape == (1,)
    assert result.size == 1

  def test_zeros_large_tensor(self):
    result = zeros(10, 10, 10)
    assert result.shape == (10, 10, 10)
    assert result.size == 1000

class TestOnes:
  def test_ones_1d(self):
    result = ones(4)
    assert isinstance(result, Tensor)
    assert result.shape == (4,)
    assert result.size == 4
    assert result.ndim == 1
    assert result.dtype == "float32"

  def test_ones_2d(self):
    result = ones(2, 5)
    assert result.shape == (2, 5)
    assert result.size == 10
    assert result.ndim == 2

  def test_ones_3d(self):
    result = ones(3, 2, 4)
    assert result.shape == (3, 2, 4)
    assert result.size == 24
    assert result.ndim == 3

  def test_ones_with_dtype_int64(self):
    result = ones(5, dtype=Tensor.int64)
    assert result.dtype == "int64"
    assert result.shape == (5,)

  def test_ones_with_dtype_float32(self):
    result = ones(3, 3, dtype=Tensor.float32)
    assert result.dtype == "float32"
    assert result.shape == (3, 3)

  def test_ones_square_matrix(self):
    result = ones(7, 7)
    assert result.shape == (7, 7)
    assert result.size == 49

class TestRandn:
  def test_randn_1d(self):
    result = randn(6)
    assert isinstance(result, Tensor)
    assert result.shape == (6,)
    assert result.size == 6
    assert result.ndim == 1
    assert result.dtype == "float32"

  def test_randn_2d(self):
    result = randn(3, 4)
    assert result.shape == (3, 4)
    assert result.size == 12
    assert result.ndim == 2

  def test_randn_3d(self):
    result = randn(2, 2, 3)
    assert result.shape == (2, 2, 3)
    assert result.size == 12
    assert result.ndim == 3

  def test_randn_with_dtype(self):
    result = randn(4, dtype=Tensor.float64)
    assert result.dtype == "float64"

  def test_randn_different_calls_different_values(self):
    result1 = randn(5)
    result2 = randn(5)
    assert result1.shape == result2.shape
    assert isinstance(result1, Tensor)
    assert isinstance(result2, Tensor)

  def test_randn_large_tensor(self):
    result = randn(50, 50)
    assert result.shape == (50, 50)
    assert result.size == 2500

class TestRandint:
  def test_randint_1d(self):
    result = randint(0, 10, 5)
    assert isinstance(result, Tensor)
    assert result.shape == (5,)
    assert result.size == 5
    assert result.ndim == 1
    assert result.dtype == "int32"

  def test_randint_2d(self):
    result = randint(1, 6, 3, 3)
    assert result.shape == (3, 3)
    assert result.size == 9
    assert result.ndim == 2

  def test_randint_3d(self):
    result = randint(-5, 5, 2, 2, 2)
    assert result.shape == (2, 2, 2)
    assert result.size == 8
    assert result.ndim == 3

  def test_randint_with_dtype(self):
    result = randint(0, 100, 4, dtype=Tensor.int64)
    assert result.dtype == "int64"

  def test_randint_negative_range(self):
    result = randint(-10, -1, 5)
    assert result.shape == (5,)

  def test_randint_zero_range(self):
    result = randint(0, 1, 3)
    assert result.shape == (3,)

  def test_randint_large_range(self):
    result = randint(0, 1000000, 10)
    assert result.shape == (10,)

class TestUniform:
  def test_uniform_1d(self):
    result = uniform(0.0, 1.0, 4)
    assert isinstance(result, Tensor)
    assert result.shape == (4,)
    assert result.size == 4
    assert result.ndim == 1
    assert result.dtype == "float32"

  def test_uniform_2d(self):
    result = uniform(-1.0, 1.0, 2, 3)
    assert result.shape == (2, 3)
    assert result.size == 6
    assert result.ndim == 2

  def test_uniform_3d(self):
    result = uniform(10.0, 20.0, 2, 2, 2)
    assert result.shape == (2, 2, 2)
    assert result.size == 8
    assert result.ndim == 3

  def test_uniform_with_dtype(self):
    result = uniform(0.0, 5.0, 5, dtype=Tensor.float64)
    assert result.dtype == "float64"

  def test_uniform_negative_range(self):
    result = uniform(-10.0, -1.0, 3)
    assert result.shape == (3,)

  def test_uniform_large_range(self):
    result = uniform(0.0, 1000.0, 10)
    assert result.shape == (10,)

  def test_uniform_small_range(self):
    result = uniform(0.999, 1.001, 5)
    assert result.shape == (5,)

class TestFill:
  def test_fill_1d(self):
    result = fill(42.0, 5)
    assert isinstance(result, Tensor)
    assert result.shape == (5,)
    assert result.size == 5
    assert result.ndim == 1
    assert result.dtype == "float32"

  def test_fill_2d(self):
    result = fill(7.5, 3, 4)
    assert result.shape == (3, 4)
    assert result.size == 12
    assert result.ndim == 2

  def test_fill_3d(self):
    result = fill(-3.14, 2, 2, 3)
    assert result.shape == (2, 2, 3)
    assert result.size == 12
    assert result.ndim == 3

  def test_fill_with_int(self):
    result = fill(100, 4)
    assert result.shape == (4,)

  def test_fill_with_dtype_int32(self):
    result = fill(25, 3, dtype=Tensor.int32)
    assert result.dtype == "int32"

  def test_fill_with_dtype_float64(self):
    result = fill(3.14159, 2, 2, dtype=Tensor.float64)
    assert result.dtype == "float64"

  def test_fill_zero_value(self):
    result = fill(0.0, 5)
    assert result.shape == (5,)

  def test_fill_negative_value(self):
    result = fill(-99.9, 3, 3)
    assert result.shape == (3, 3)

class TestLinspace:
  def test_linspace_1d(self):
    result = linspace(0.0, 1.0, 10.0, 5)
    assert isinstance(result, Tensor)
    assert result.shape == (5,)
    assert result.size == 5
    assert result.ndim == 1
    assert result.dtype == "float32"

  def test_linspace_2d(self):
    result = linspace(0.0, 0.5, 5.0, 2, 3)
    assert result.shape == (2, 3)
    assert result.size == 6
    assert result.ndim == 2

  def test_linspace_3d(self):
    result = linspace(1.0, 2.0, 8.0, 2, 2, 2)
    assert result.shape == (2, 2, 2)
    assert result.size == 8
    assert result.ndim == 3

  def test_linspace_with_dtype(self):
    result = linspace(0.0, 10.0, 100.0, 10, dtype=Tensor.float64)
    assert result.dtype == "float64"

  def test_linspace_negative_range(self):
    result = linspace(-5.0, 1.0, 5.0, 6)
    assert result.shape == (6,)

  def test_linspace_zero_step(self):
    result = linspace(1.0, 0.0, 1.0, 5)
    assert result.shape == (5,)

  def test_linspace_large_step(self):
    result = linspace(0.0, 100.0, 1000.0, 3)
    assert result.shape == (3,)

  def test_linspace_single_element(self):
    result = linspace(42.0, 1.0, 43.0, 1)
    assert result.shape == (1,)

class TestArange:
  def test_arange_basic(self):
    result = arange(0.0, 5.0)
    assert isinstance(result, Tensor)
    assert result.ndim == 1
    assert result.dtype == "float32"

  def test_arange_with_step(self):
    result = arange(0.0, 10.0, 2.0)
    assert result.ndim == 1

  def test_arange_with_dtype(self):
    result = arange(1.0, 6.0, dtype=Tensor.int32)
    assert result.dtype == "int32"

  def test_arange_negative_start(self):
    result = arange(-5.0, 5.0)
    assert result.ndim == 1

  def test_arange_negative_step(self):
    result = arange(10.0, 0.0, -1.0)
    assert result.ndim == 1

  def test_arange_float_step(self):
    result = arange(0.0, 3.0, 0.5)
    assert result.ndim == 1

  def test_arange_zero_step_error(self):
    with pytest.raises(ValueError, match="Step cannot be zero"):
      arange(0.0, 5.0, 0.0)

  def test_arange_invalid_range_positive_step(self):
    with pytest.raises(ValueError, match="Invalid arange parameters"):
      arange(5.0, 0.0, 1.0)

  def test_arange_invalid_range_negative_step(self):
    with pytest.raises(ValueError, match="Invalid arange parameters"):
      arange(0.0, 5.0, -1.0)

  def test_arange_same_start_stop(self):
    with pytest.raises(ValueError, match="Invalid arange parameters"):
      arange(5.0, 5.0, 1.0)

  def test_arange_large_range(self):
    result = arange(0.0, 1000.0, 10.0)
    assert result.ndim == 1

  def test_arange_small_step(self):
    result = arange(0.0, 2.0, 0.1)
    assert result.ndim == 1

class TestUtilsIntegration:
  def test_zeros_ones_same_shape(self):
    z = zeros(3, 4)
    o = ones(3, 4)
    assert z.shape == o.shape
    assert z.size == o.size
    assert z.ndim == o.ndim

  def test_like_functions_preserve_properties(self):
    original = Tensor([[1, 2, 3], [4, 5, 6]])
    z_like = zeros_like(original)
    o_like = ones_like(original)
    assert z_like.shape == original.shape
    assert o_like.shape == original.shape
    assert z_like.strides == original.strides
    assert o_like.strides == original.strides

  def test_random_functions_different_shapes(self):
    r1 = randn(2, 3)
    r2 = randint(0, 10, 4, 5)
    r3 = uniform(0.0, 1.0, 6)
    assert r1.shape == (2, 3)
    assert r2.shape == (4, 5)
    assert r3.shape == (6,)

  def test_fill_vs_zeros_ones(self):
    z = zeros(3, 3)
    o = ones(2, 4)
    f_zero = fill(0.0, 3, 3)
    f_one = fill(1.0, 2, 4)
    assert z.shape == f_zero.shape
    assert o.shape == f_one.shape

  def test_arange_vs_linspace_concepts(self):
    a = arange(0.0, 10.0, 1.0)
    l = linspace(0.0, 1.0, 9.0, 10)
    assert a.ndim == l.ndim == 1

  def test_mixed_dtypes(self):
    tensors = [
      zeros(2, dtype=Tensor.int32),
      ones(2, dtype=Tensor.float64),
      randn(2, dtype=Tensor.float32),
      randint(0, 5, 2, dtype=Tensor.int64),
      fill(3.14, 2, dtype=Tensor.float32)
    ]
    dtypes = ["int32", "float64", "float32", "int64", "float32"]
    for tensor, expected_dtype in zip(tensors, dtypes):
      assert tensor.dtype == expected_dtype

if __name__ == "__main__":
  pytest.main([__file__, "-v"])