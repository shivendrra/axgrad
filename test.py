#!/usr/bin/env python3
"""
Test cases for Tensor indexing and slicing functionality
Tests all __getitem__, __setitem__, and __iter__ operations
"""

from axgrad import Tensor

def test_tensor_indexing():
  print("=== Testing Tensor Indexing & Slicing ===\n")
  
  # Test 1: 1-D Tensor indexing
  print("Test 1: 1-D Tensor indexing")
  tensor_1d = Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
  print(f"Original 1-D tensor: {tensor_1d}")
  
  # Single element access
  elem = tensor_1d[0]
  print(f"tensor_1d[0] = {elem} (type: {type(elem)})")
  assert isinstance(elem, float), f"Expected float, got {type(elem)}"
  assert elem == 1.0, f"Expected 1.0, got {elem}"
  
  # Negative indexing
  elem_neg = tensor_1d[-1]
  print(f"tensor_1d[-1] = {elem_neg} (type: {type(elem_neg)})")
  assert elem_neg == 5.0, f"Expected 5.0, got {elem_neg}"
  
  # Single element assignment
  tensor_1d[1] = 10.0
  elem_modified = tensor_1d[1]
  print(f"After tensor_1d[1] = 10.0: tensor_1d[1] = {elem_modified}")
  assert elem_modified == 10.0, f"Expected 10.0, got {elem_modified}"
  print("✓ 1-D indexing passed\n")
  
  # Test 2: 2-D Tensor indexing
  print("Test 2: 2-D Tensor indexing")
  tensor_2d = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
  print(f"Original 2-D tensor: {tensor_2d}")
  
  # Row access (should return list)
  row = tensor_2d[0]
  print(f"tensor_2d[0] = {row} (type: {type(row)})")
  assert isinstance(row, list), f"Expected list, got {type(row)}"
  assert row == [1.0, 2.0, 3.0], f"Expected [1.0, 2.0, 3.0], got {row}"
  
  # Single element access with tuple
  elem = tensor_2d[1, 2]
  print(f"tensor_2d[1, 2] = {elem} (type: {type(elem)})")
  assert isinstance(elem, float), f"Expected float, got {type(elem)}"
  assert elem == 6.0, f"Expected 6.0, got {elem}"
  
  # Single element assignment with tuple
  tensor_2d[0, 0] = 99.0
  elem_modified = tensor_2d[0, 0]
  print(f"After tensor_2d[0, 0] = 99.0: tensor_2d[0, 0] = {elem_modified}")
  assert elem_modified == 99.0, f"Expected 99.0, got {elem_modified}"
  
  # Row assignment
  tensor_2d[2] = [70.0, 80.0, 90.0]
  row_modified = tensor_2d[2]
  print(f"After tensor_2d[2] = [70.0, 80.0, 90.0]: tensor_2d[2] = {row_modified}")
  assert row_modified == [70.0, 80.0, 90.0], f"Expected [70.0, 80.0, 90.0], got {row_modified}"
  print("✓ 2-D indexing passed\n")
  
  # Test 3: 3-D Tensor indexing
  print("Test 3: 3-D Tensor indexing")
  tensor_3d = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
  print(f"Original 3-D tensor: {tensor_3d}")
  
  # 2-D slice access
  slice_2d = tensor_3d[0]
  print(f"tensor_3d[0] = {slice_2d} (type: {type(slice_2d)})")
  assert isinstance(slice_2d, list), f"Expected list, got {type(slice_2d)}"
  assert slice_2d == [[1.0, 2.0], [3.0, 4.0]], f"Expected [[1.0, 2.0], [3.0, 4.0]], got {slice_2d}"
  
  # 1-D slice access
  slice_1d = tensor_3d[1, 0]
  print(f"tensor_3d[1, 0] = {slice_1d} (type: {type(slice_1d)})")
  assert isinstance(slice_1d, float), f"Expected float, got {type(slice_1d)}"
  
  # Single element access
  elem = tensor_3d[1, 1, 0]
  print(f"tensor_3d[1, 1, 0] = {elem} (type: {type(elem)})")
  assert isinstance(elem, float), f"Expected float, got {type(elem)}"
  assert elem == 7.0, f"Expected 7.0, got {elem}"
  
  # Single element assignment
  tensor_3d[0, 0, 1] = 20.0
  elem_modified = tensor_3d[0, 0, 1]
  print(f"After tensor_3d[0, 0, 1] = 20.0: tensor_3d[0, 0, 1] = {elem_modified}")
  assert elem_modified == 20.0, f"Expected 20.0, got {elem_modified}"
  print("✓ 3-D indexing passed\n")
  
  # Test 4: Iterator functionality
  print("Test 4: Iterator functionality")
  tensor_iter = Tensor([10.0, 20.0, 30.0])
  print(f"Original tensor for iteration: {tensor_iter}")
  
  # Test iteration over 1-D tensor
  iter_values = list(tensor_iter)
  print(f"Iteration values: {iter_values}")
  assert iter_values == [10.0, 20.0, 30.0], f"Expected [10.0, 20.0, 30.0], got {iter_values}"
  assert all(isinstance(v, float) for v in iter_values), "All iteration values should be floats"
  
  # Test iteration over 2-D tensor
  tensor_2d_iter = Tensor([[1.0, 2.0], [3.0, 4.0]])
  iter_2d_values = list(tensor_2d_iter)
  print(f"2-D iteration values: {iter_2d_values}")
  assert iter_2d_values == [[1.0, 2.0], [3.0, 4.0]], f"Expected [[1.0, 2.0], [3.0, 4.0]], got {iter_2d_values}"
  print("✓ Iterator functionality passed\n")
  
  # Test 5: Edge cases and error handling
  print("Test 5: Edge cases and error handling")
  
  # Index out of bounds
  try:
    _ = tensor_1d[10]
    assert False, "Should have raised IndexError"
  except IndexError as e:
    print(f"✓ Out of bounds index correctly raised: {e}")
  
  # Too many indices
  try:
    _ = tensor_1d[0, 1]
    assert False, "Should have raised IndexError"
  except IndexError as e:
    print(f"✓ Too many indices correctly raised: {e}")
  
  # 0-D tensor indexing (should fail)
  try:
    scalar_tensor = Tensor([5.0])
    scalar_tensor.ndim = 0  # Simulate 0-D tensor
    _ = scalar_tensor[0]
    assert False, "Should have raised TypeError"
  except TypeError as e:
    print(f"✓ 0-D tensor indexing correctly raised: {e}")
  
  print("✓ Edge cases passed\n")
  
  # Test 6: Assignment with different value types
  print("Test 6: Assignment with different value types")
  test_tensor = Tensor([1.0, 2.0, 3.0])
  
  # Assign Tensor value (scalar)
  scalar_tensor = Tensor([42.0])
  test_tensor[0] = scalar_tensor
  assert test_tensor[0] == 42.0, f"Expected 42.0, got {test_tensor[0]}"
  print("✓ Tensor scalar assignment passed")
  
  # Assign list value (single element)
  test_tensor[1] = [55.0]
  assert test_tensor[1] == 55.0, f"Expected 55.0, got {test_tensor[1]}"
  print("✓ List scalar assignment passed")
  
  # Assign float value
  test_tensor[2] = 77.0
  assert test_tensor[2] == 77.0, f"Expected 77.0, got {test_tensor[2]}"
  print("✓ Float assignment passed\n")
  
  print("🎉 All tensor indexing and slicing tests passed!")

if __name__ == "__main__":
  # Note: This assumes the Tensor class is imported
  # from your_module import Tensor
  test_tensor_indexing()