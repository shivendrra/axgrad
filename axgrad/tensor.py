from ctypes import c_float, c_size_t, c_int, c_bool
from typing import *

from ._core import CTensor, lib, DType
from .helpers import ShapeHelp, DtypeHelp
from .autograd.functions import *
from .ops.binary import register_binary_ops
from .ops.functional import register_functional_ops

int8, int16, int32, int64, long = "int8", "int16", "int32", "int64", "long"
float32, float64, double = "float32", "float64", "double"
uint8, uint16, uint32, uint64 = "uint8", "uint16", "uint32", "uint64"
boolean = "bool"

class Slice:
  def __init__(self, parent_tensor, row_index, shape, size, strides):
    self.parent_tensor, self.row_index, self.shape, self.size, self.strides, self.ndim = parent_tensor, row_index, shape, size, strides, len(shape)
  def __getitem__(self, sub_key):
    if isinstance(sub_key, int):
      if sub_key < 0: sub_key += self.shape[0]
      if sub_key < 0 or sub_key >= self.shape[0]: raise IndexError(f"Index {sub_key} out of bounds")
      if self.ndim == 1:
        indices = [self.row_index, sub_key]
        indices_ctypes = (c_int * len(indices))(*indices)
        return lib.get_item_tensor(self.parent_tensor.data, indices_ctypes)
      else: return Slice(self.parent_tensor, self.row_index, self.shape[1:], self.size // self.shape[0], self.strides[1:])
    elif isinstance(sub_key, tuple):
      indices = [self.row_index] + list(sub_key)
      if len(indices) > self.parent_tensor.ndim: raise IndexError(f"Too many indices")
      indices += [0] * (self.parent_tensor.ndim - len(indices))
      indices_ctypes = (c_int * len(indices))(*indices)
      return lib.get_item_tensor(self.parent_tensor.data, indices_ctypes)
    else: raise TypeError("Index must be int or tuple of ints")

  def __setitem__(self, sub_key, value):
    if isinstance(sub_key, int):
      if sub_key < 0: sub_key += self.shape[0]
      if sub_key < 0 or sub_key >= self.shape[0]: raise IndexError(f"Index {sub_key} out of bounds")
      indices = [self.row_index, sub_key]
      if len(indices) < self.parent_tensor.ndim: indices += [0] * (self.parent_tensor.ndim - len(indices))
      if isinstance(value, Tensor):
        if value.size != 1: raise ValueError("Can only assign scalar tensors")
        value = value.tolist()
      indices_ctypes = (c_int * len(indices))(*indices)
      lib.set_item_tensor(self.parent_tensor.data, indices_ctypes, c_float(value))
    elif isinstance(sub_key, tuple):
      indices = [self.row_index] + list(sub_key)
      if len(indices) > self.parent_tensor.ndim: raise IndexError(f"Too many indices")
      indices += [0] * (self.parent_tensor.ndim - len(indices))
      if isinstance(value, Tensor):
        if value.size != 1: raise ValueError("Can only assign scalar tensors")
        value = value.tolist()      
      indices_ctypes = (c_int * len(indices))(*indices)
      lib.set_item_tensor(self.parent_tensor.data, indices_ctypes, c_float(value))
    else: raise TypeError("Index must be int or tuple of ints")

class Tensor:
  int8, int16, int32, int64, long, float32, float64, double, uint8, uint16, uint32, uint64, boolean = int8, int16, int32, int64, long, float32, float64, double, uint8, uint16, uint32, uint64, boolean
  
  def __init__(self, data: Union[List[Any], int, float], dtype: str=float32, requires_grad: bool=False):
    if isinstance(data, CTensor): self.data, self.shape, self.size, self.ndim, self.strides, self.dtype = data, (), 0, 0, [], dtype or "float32"
    elif isinstance(data, Tensor): self.data, self.shape, self.dtype, self.size, self.ndim, self.strides = data.data, data.shape, dtype or data.dtype, data.size, data.ndim, data.strides
    else:
      data, shape = ShapeHelp.flatten([data] if isinstance(data, (int, float)) else data), tuple(ShapeHelp.get_shape(data))
      self.size, self.ndim, self.dtype, self.shape, self.strides = len(data), len(shape), dtype or "float32", shape, ShapeHelp.get_strides(shape)
      self._data_ctypes, self._shape_ctypes = (c_float * self.size)(*data.copy()), (c_int * self.ndim)(*shape)
      self.data = lib.create_tensor(self._data_ctypes, c_size_t(self.ndim), self._shape_ctypes, c_size_t(self.size), c_int(DtypeHelp._parse_dtype(self.dtype)))
    self.requires_grad, self.hooks, self.grad_fn, self.grad = requires_grad, [], None, None

  def backward(self, gradient=None):
    assert self.ndim == 0 or (self.ndim == 1 and self.size == 1), "backward can only be called for scalar tensors"
    if gradient is None: gradient = Tensor([1.0], dtype=self.dtype)
    visited, topo_order = set(), []

    def build_topo(v):
      if v in visited or not v.requires_grad: return
      visited.add(v)
      if v.grad_fn: [build_topo(inp) for inp in v.grad_fn.input if isinstance(inp, Tensor)]
      topo_order.append(v)

    build_topo(self)
    self.grad = gradient if self.grad is None else self.grad + gradient

    for tensor in reversed(topo_order):
      if tensor.grad_fn and tensor.grad is not None:
        grads = tensor.grad_fn.backward(tensor.grad)
        for inp, grad in zip(tensor.grad_fn.input, grads):
          if isinstance(inp, Tensor) and inp.requires_grad:
            inp.grad = grad if inp.grad is None else inp.grad + grad

  def __setattr__(self, name, value):
    if name == "grad": [setattr(self, "_temp_value", hook(getattr(self, "_temp_value", value))) for hook in self.hooks]; value = getattr(self, "_temp_value", value)
    super().__setattr__(name, value)
  def register_hook(self, function): self.hooks.append(function)
  def __str__(self): return (lib.print_tensor(self.data), "")[1]
  def astype(self, dtype: DType) -> "Tensor":
    out = Tensor(lib.cast_tensor(self.data, c_int(DtypeHelp._parse_dtype(dtype))).contents, requires_grad=self.requires_grad)
    out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
    return out

  def __getitem__(self, key):
    if self.ndim == 0: raise TypeError("0-d tensor cannot be indexed")
    if isinstance(key, int):
      if key < 0: key += self.shape[0]
      if key < 0 or key >= self.shape[0]: raise IndexError(f"Index {key} out of bounds for dimension 0 with size {self.shape[0]}")

      if self.ndim == 1:
        indices = [key]
        indices_ctypes = (c_int * len(indices))(*indices)
        return lib.get_item_tensor(self.data, indices_ctypes)
      else:
        new_shape = self.shape[1:]
        new_size = self.size // self.shape[0]
        new_strides = self.strides[1:]
        return Slice(self, key, new_shape, new_size, new_strides)

    elif isinstance(key, tuple):
      if len(key) > self.ndim: raise IndexError(f"Too many indices for tensor: got {len(key)}, expected {self.ndim}")
      indices = list(key) + [0] * (self.ndim - len(key))
      indices_ctypes = (c_int * len(indices))(*indices)
      return lib.get_item_tensor(self.data, indices_ctypes)
    else: raise TypeError("Index must be int or tuple of ints")

  def __setitem__(self, key, value):
    if self.ndim == 0: raise TypeError("0-d tensor cannot be indexed")
    if isinstance(key, int):
      if key < 0: key += self.shape[0]
      if key < 0 or key >= self.shape[0]: raise IndexError(f"Index {key} out of bounds for dimension 0 with size {self.shape[0]}")
      
      if self.ndim == 1:
        indices = [key]
        if isinstance(value, Tensor):
          if value.size != 1: raise ValueError("Can only assign scalar tensors")
          value = value.tolist()
        indices_ctypes = (c_int * len(indices))(*indices)
        lib.set_item_tensor(self.data, indices_ctypes, c_float(value))
      else:
        if isinstance(value, (list, tuple)):
          flat_value = ShapeHelp.flatten(value)
          expected_size = self.size // self.shape[0]
          if len(flat_value) != expected_size: raise ValueError(f"Cannot assign {len(flat_value)} values to slice of size {expected_size}")
          
          for i, val in enumerate(flat_value):
            linear_idx = key * self.strides[0] + i
            temp_indices = []
            temp_linear = linear_idx
            for dim in range(self.ndim - 1, -1, -1):
              temp_indices.insert(0, temp_linear % self.shape[dim])
              temp_linear //= self.shape[dim]
            indices_ctypes = (c_int * len(temp_indices))(*temp_indices)
            lib.set_item_tensor(self.data, indices_ctypes, c_float(val))
        else:
          raise ValueError("Cannot assign scalar to multi-dimensional slice")
    elif isinstance(key, tuple):
      if len(key) > self.ndim: raise IndexError(f"Too many indices for tensor: got {len(key)}, expected {self.ndim}")
      indices = list(key) + [0] * (self.ndim - len(key))
      if isinstance(value, Tensor):
        if value.size != 1: raise ValueError("Can only assign scalar tensors")
        value = value.tolist()
      indices_ctypes = (c_int * len(indices))(*indices)
      lib.set_item_tensor(self.data, indices_ctypes, c_float(value))
    else: raise TypeError("Index must be int or tuple of ints")

  def __iter__(self):
    if self.ndim == 0: raise TypeError("Iteration over 0-d tensor")
    for i in range(self.shape[0]):
      if self.ndim == 1: yield self[i]
      else:
        row_data = []
        offset = i * self.strides[0]
        row_size = self.size // self.shape[0]
        for j in range(row_size):
          linear_idx, indices = offset + j, []
          temp_idx = linear_idx
          for dim in range(self.ndim - 1, 0, -1):
            indices.insert(0, temp_idx % self.shape[dim])
            temp_idx //= self.shape[dim]
          indices.insert(0, i)
          indices_ctypes = (c_int * len(indices))(*indices)
          value = lib.get_item_tensor(self.data, indices_ctypes)
          row_data.append(value)
        yield ShapeHelp.reshape_list(row_data, self.shape[1:])

  def tolist(self) -> List[Any]:
    data_ptr = lib.out_data(self.data)
    data_tensor = [data_ptr[i] for i in range(self.size)]
    if self.ndim == 0: return data_tensor[0]
    elif self.ndim == 1: return data_tensor
    else: return ShapeHelp.reshape_list(data_tensor, self.shape)

  def transpose(self) -> "Tensor":
    assert self.ndim <= 3, ".transpose() ops limited to 3-d tensor"
    out = Tensor(lib.transpose_tensor(self.data).contents, self.dtype, self.requires_grad)
    out.shape, out.size, out.ndim = tuple(ShapeHelp.transpose_shape(self.shape)), self.size, self.ndim
    out.strides = ShapeHelp.get_strides(out.shape)
    if self.requires_grad: out.grad_fn = TransposeBackwards(self)
    return out

  def flatten(self) -> "Tensor":
    out = Tensor(lib.flatten_tensor(self.data).contents, self.dtype, self.requires_grad)
    out.shape, out.size, out.ndim = (self.size, ), self.size, 1
    out.strides = ShapeHelp.get_strides(out.shape)
    if self.requires_grad: out.grad_fn = FlatBackwards(self)
    return out

  def reshape(self, new_shape: Union[list, tuple]) -> "Tensor":
    if isinstance(new_shape, tuple): new_shape = list(new_shape)
    new_size, ndim = 1, len(new_shape)
    for dim in new_shape: new_size *= dim
    if new_size != self.size: raise ValueError(f"Cannot reshape Tensor of size {self.size} into shape {new_shape}")
    out = Tensor(lib.reshape_tensor(self.data, (c_int * ndim)(*new_shape), c_int(ndim)).contents, self.dtype, self.requires_grad)
    out.shape, out.size, out.ndim, out.strides = tuple(new_shape), self.size, ndim, ShapeHelp.get_strides(new_shape)
    if self.requires_grad: out.grad_fn = ReshapeBackwards(self)
    return out

  def sum(self, axis: int = -1, keepdims: bool = False) -> "Tensor":
    out = Tensor(lib.sum_tensor(self.data, c_int(axis), c_bool(keepdims)).contents, self.dtype, self.requires_grad)
    if axis == -1: out.shape, out.size, out.ndim = (1,) if keepdims else (), 1, 1 if keepdims else 0
    else:
      new_shape = list(self.shape)
      if keepdims: new_shape[axis] = 1
      else: new_shape.pop(axis)
      out.shape = tuple(new_shape)
      out.size, out.ndim, out.strides = 1 if not new_shape else eval('*'.join(map(str, new_shape))), len(new_shape), ShapeHelp.get_strides(out.shape) if out.shape else []
    if self.requires_grad: out.grad_fn = SumBackwards(self, axis, keepdims)
    return out

  def __neg__(self) -> "Tensor":
    result_pointer = lib.neg_tensor(self.data).contents
    out = Tensor(result_pointer, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if out.requires_grad: out.grad_fn = NegBackwards(self)
    return (setattr(out, "grad", self.grad), setattr(out, "hooks", self.hooks), setattr(out, "grad_fn", self.grad_fn), out)[3] if self.requires_grad else out

  def sign(self) -> "Tensor":
    result_pointer = lib.sign_tensor(self.data).contents
    out = Tensor(result_pointer, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    return (setattr(out, "grad", self.grad), setattr(out, "hooks", self.hooks), setattr(out, "grad_fn", self.grad_fn), out)[3] if self.requires_grad else out

  def log(self) -> "Tensor":
    result_pointer = lib.log_tensor(self.data).contents
    out = Tensor(result_pointer, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = LogBackwards(self)
    return out

  def exp(self) -> "Tensor":
    result_pointer = lib.exp_tensor(self.data).contents
    out = Tensor(result_pointer, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = ExpBackwards(self)
    return out

  def abs(self) -> "Tensor":
    result_pointer = lib.abs_tensor(self.data).contents
    out = Tensor(result_pointer, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = AbsBackwards(self)
    return out

  def sqrt(self) -> "Tensor":
    result_pointer = lib.sqrt_tensor(self.data).contents
    out = Tensor(result_pointer, self.dtype, self.requires_grad)
    out.shape, out.ndim, out.size, out.strides = self.shape, self.ndim, self.size, self.strides
    if self.requires_grad: out.grad_fn = SqrtBackwards(self)
    return out

register_binary_ops()
register_functional_ops()