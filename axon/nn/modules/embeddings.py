from ..module import Module
from ..parameter import Parameter
from ...tensor import tensor

class Embedding(Module):
  def __init__(self, n_embd, d_embd, pad_idx=None, max_norm=None, norm_type=2.0):
    super().__init__()
    self.n_embd = n_embd
    self.d_embd = d_embd
    self.pad_idx = pad_idx
    self.max_norm = max_norm
    self.norm_type = norm_type
    
    self.wei = Parameter((n_embd, d_embd))
    if self.pad_idx is not None:
      self.wei.data[pad_idx] = [0.0] * d_embd

  def forward(self, _in):
    _in = _in if isinstance(_in, tensor) else tensor(_in)
    indices = _in.data
    embedded = [self.wei.data[idx] for idx in indices]
    embedded = tensor(embedded, child=(_in, self.wei))

    if self.max_norm is not None:
      self._normalize(embedded)

    return embedded
  
  def _normalize(self, embedded_tensor):
    for i, vec in enumerate(embedded_tensor.data):
      norm = sum([x**self.norm_type for x in vec])**(1.0 / self.norm_type)
      scale = self.max_norm / max(norm, 1e-7)
      if scale < 1:
        embedded_tensor.data[i] = [x * scale for x in vec]
  
  def __call__(self, _in:tensor) -> tensor:
    return self.forward(_in)
  
  def parameters(self) -> tensor:
    return [self.wei]
  
  def __repr__(self) -> str:
    return f"<Embedding(num_embeddings={self.n_embd}, embedding_dim={self.d_embd})>"