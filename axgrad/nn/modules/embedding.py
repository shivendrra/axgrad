from ...tensor import Tensor
from ..parameter import Parameter
from ..module import Module
from ..._core import lib
from ...helpers import ShapeHelp, DtypeHelp
from ...autograd.functions import *

from ctypes import c_int, c_size_t, c_float
import math

class Embedding(Module):
  def __init__(self, n_embed: int, d_embed: int, max_norm: float=None, norm_type: float=2.0, dtype: str="float32"):
    super().__init__()
    self.n_embed, self.d_embed, self.max_norm, self.norm_type, self.dtype = n_embed, d_embed, max_norm, norm_type, dtype

    std = math.sqrt(1.0 / d_embed)
    self.weight = Parameter((n_embed, d_embed), dtype)
    self.weight.set_name("weight")
    self._init_uniform_weights(self.weight, -std, std)

  def _init_uniform_weights(self, param: Parameter, low: float, high: float):
    uniform_data = lib.uniform_tensor(c_int(int(low * 10)), c_int(int(high * 10)), (c_int * param.ndim)(*param.shape), c_size_t(param.size),
      c_size_t(param.ndim), c_int(DtypeHelp._parse_dtype(param.dtype))).contents
    param.data = uniform_data

  def _apply_norm_clipping(self, indices: Tensor):
    if self.max_norm is None: return

    if indices.ndim == 0: unique_indices = [int(indices.item())]
    else:
      flat_indices = indices.reshape((-1,))
      unique_indices = list(set(int(flat_indices.data[i]) for i in range(flat_indices.size)))

    for idx in unique_indices:
      if 0 <= idx < self.n_embed:
        embedding_vec = self.weight.data[idx:idx+1]
        norm_data = lib.norm_tensor(embedding_vec, c_float(self.norm_type), c_int(self.d_embed)).contents
        norm_val = norm_data[0]

        if norm_val > self.max_norm:
          scale = self.max_norm / norm_val
          lib.scale_tensor_inplace(embedding_vec, c_float(scale), c_int(self.d_embed))

  def forward(self, input: Tensor) -> Tensor:
    assert input.dtype in ["int32", "int64"], f"Input must be integer indices, got {input.dtype}"
    
    if self.max_norm is not None: self._apply_norm_clipping(input)
    
    if input.ndim == 0:
      idx = int(input.item())
      assert 0 <= idx < self.n_embed, f"Index {idx} out of range [0, {self.n_embed})"
      result_data = lib.embedding_lookup(self.weight.data, c_int(idx), c_int(self.d_embed)).contents
      out = Tensor(result_data, self.dtype, requires_grad=self.weight.requires_grad)
      out.shape, out.size, out.ndim = (self.d_embed,), self.d_embed, 1
      out.strides = ShapeHelp.get_strides(out.shape)
      if out.requires_grad: out.grad_fn = EmbeddingBackwards(self.weight, input)
      return out
    
    original_shape = input.shape
    flat_input = input.reshape((-1,))
    batch_size = flat_input.size

    output_data = lib.embedding_batch_lookup(self.weight.data, flat_input.data, c_int(batch_size), c_int(self.n_embed), c_int(self.d_embed)).contents
    out_shape = original_shape + (self.d_embed,)
    out_size = 1
    for dim in out_shape: out_size *= dim

    out = Tensor(output_data, self.dtype, requires_grad=self.weight.requires_grad)
    out.shape, out.size, out.ndim = out_shape, out_size, len(out_shape)
    out.strides = ShapeHelp.get_strides(out.shape)
    if out.requires_grad: out.grad_fn = EmbeddingBackwards(self.weight, input)
    return out

  def __call__(self, input: Tensor) -> Tensor: return self.forward(input)
  def __repr__(self) -> str: return f"Embedding(n_embed={self.n_embed}, d_embed={self.d_embed})"
  def inner_repr(self) -> str: return f"n_embed={self.n_embed}, d_embed={self.d_embed}"
  def parameters(self): yield self, "weight", self.weight
  def zero_grad(self):
    for _, _, param in self.parameters(): param.zero_grad()