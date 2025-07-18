from ...tensor import Tensor
from ..parameter import Parameter
from ..module import Module
from ..._core import lib
from ...helpers import ShapeHelp, DtypeHelp
from ...autograd.nn import EmbeddingBackwards

from ctypes import c_int, c_size_t, c_float
import math

class Embedding(Module):
  def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int=None, max_norm: float=None, norm_type: float=2.0, scale_grad_by_freq: bool=False, sparse: bool=False, dtype: str="float32"):
    super().__init__()
    self.num_embeddings, self.embedding_dim, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse, self.dtype = num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, dtype

    self.weight = Parameter((num_embeddings, embedding_dim), dtype)
    self.weight.set_name("weight")
    self._init_weights()

  def _init_weights(self):
    std = math.sqrt(1.0 / self.embedding_dim)
    uniform_data = lib.uniform_tensor(c_int(int(-std * 10)), c_int(int(std * 10)), (c_int * self.weight.ndim)(*self.weight.shape), c_size_t(self.weight.size), c_size_t(self.weight.ndim), c_int(DtypeHelp._parse_dtype(self.weight.dtype))).contents
    self.weight.data = uniform_data
    if self.padding_idx is not None: self._zero_padding_idx()

  def _zero_padding_idx(self):
    if 0 <= self.padding_idx < self.num_embeddings:
      for i in range(self.embedding_dim):
        indices_ctypes = (c_int * 2)(self.padding_idx, i)
        lib.set_item_tensor(self.weight.data, indices_ctypes, c_float(0.0))

  def _apply_norm_clipping(self, indices: Tensor):
    if self.max_norm is None: return
    unique_indices = list(set(int(indices.data[i]) for i in range(indices.size))) if indices.ndim > 0 else [int(indices.item())]
    for idx in unique_indices:
      if 0 <= idx < self.num_embeddings and idx != self.padding_idx:
        embedding_vec = self.weight.data[idx:idx+1]
        norm_data = lib.norm_tensor(embedding_vec, c_float(self.norm_type), c_int(self.embedding_dim)).contents
        norm_val = norm_data[0]
        if norm_val > self.max_norm:
          scale = self.max_norm / norm_val
          lib.scale_tensor_inplace(embedding_vec, c_float(scale), c_int(self.embedding_dim))

  def _get_embedding_row(self, idx: int):
    row_data = []
    for j in range(self.embedding_dim):
      indices_ctypes = (c_int * 2)(idx, j)
      value = lib.get_item_tensor(self.weight.data, indices_ctypes)
      row_data.append(value)
    return row_data

  def forward(self, input: Tensor) -> Tensor:
    input = input if isinstance(input, Tensor) else Tensor(input, "int32", False)
    # assert input.dtype in ["int32", "int64"], f"Input must be integer indices, got {input.dtype}"
    if self.max_norm is not None: self._apply_norm_clipping(input)
    if input.ndim == 0:
      idx = int(input.item())
      assert 0 <= idx < self.num_embeddings, f"Index {idx} out of range [0, {self.num_embeddings})"
      out = Tensor(self._get_embedding_row(idx), self.dtype, requires_grad=self.weight.requires_grad)
      if out.requires_grad: out.grad_fn = EmbeddingBackwards(self.weight, input)
      return out
    original_shape, flat_input = input.shape, input.flatten()
    output_data = []    
    for i in range(flat_input.size):
      idx = int(flat_input[i])
      assert 0 <= idx < self.num_embeddings, f"Index {idx} out of range [0, {self.num_embeddings})"
      output_data.extend(self._get_embedding_row(idx))

    out_shape = original_shape + (self.embedding_dim,)
    out = Tensor(ShapeHelp.reshape_list(output_data, out_shape), self.dtype, requires_grad=self.weight.requires_grad)
    if out.requires_grad: out.grad_fn = EmbeddingBackwards(self.weight, input)
    return out

  def __call__(self, input: Tensor) -> Tensor: return self.forward(input)
  def __repr__(self) -> str: return f"Embedding({self.num_embeddings}, {self.embedding_dim})"
  def inner_repr(self) -> str: return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
  def parameters(self): yield self, "weight", self.weight
  def zero_grad(self):
    for _, _, param in self.parameters(): param.zero_grad()