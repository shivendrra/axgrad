from .loss import MSELoss, MAELoss
from ..tensor import Tensor
from .module import Module

def mse(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor: return MSELoss(reduction)(pred, target)
def mae(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor: return MAELoss(reduction)(pred, target)

class Tanh(Module):
  def __init__(self, inplace: Tensor = False):
    self.inplace = inplace
    super().__init__()
  def forward(self, x: Tensor) -> Tensor: return x.tanh()
  def __repr__(self) -> str: return f"Tanh(inplace={self.inplace})"
  def inner_repr(self) -> str: return f"inplace={self.inplace}"

class Sigmoid(Module):
  def __init__(self, inplace: Tensor = False):
    self.inplace = inplace
    super().__init__()
  def forward(self, x: Tensor) -> Tensor: return x.sigmoid()
  def __repr__(self) -> str: return f"Sigmoid(inplace={self.inplace})"
  def inner_repr(self) -> str: return f"inplace={self.inplace}"

class ReLU(Module):
  def __init__(self, inplace: Tensor = False):
    self.inplace = inplace
    super().__init__()
  def forward(self, x: Tensor) -> Tensor: return x.relu()
  def __repr__(self) -> str: return f"ReLU(inplace={self.inplace})"
  def inner_repr(self) -> str: return f"inplace={self.inplace}"

class LeakyReLU(Module):
  def __init__(self, eps: float = 1e-3, inplace: Tensor = False):
    self.inplace, self.eps = inplace, eps
    super().__init__()
  def forward(self, x: Tensor) -> Tensor: return x.leaky_relu(self.eps)
  def __repr__(self) -> str: return f"LeakyReLU(eps={self.eps}, inplace={self.inplace})"
  def inner_repr(self) -> str: return f"inplace={self.inplace}"

class ELU(Module):
  def __init__(self, alpha: float = 1e-5, inplace: Tensor = False):
    self.inplace, self.alpha = inplace, alpha
    super().__init__()
  def forward(self, x: Tensor) -> Tensor: return x.elu(self.alpha)
  def __repr__(self) -> str: return f"ELU(alpha={self.alpha} inplace={self.inplace})"
  def inner_repr(self) -> str: return f"inplace={self.inplace}"

class SiLU(Module):
  def __init__(self, inplace: Tensor = False):
    self.inplace = inplace
    super().__init__()
  def forward(self, x: Tensor) -> Tensor: return x.silu()
  def __repr__(self) -> str: return f"SiLU(inplace={self.inplace})"
  def inner_repr(self) -> str: return f"inplace={self.inplace}"

class GELU(Module):
  def __init__(self, inplace: Tensor = False):
    self.inplace = inplace
    super().__init__()
  def forward(self, x: Tensor) -> Tensor: return x.gelu()
  def __repr__(self) -> str: return f"GELU(inplace={self.inplace})"
  def inner_repr(self) -> str: return f"inplace={self.inplace}"

class Swish(Module):
  def __init__(self, beta: float = 0.5, inplace: Tensor = False):
    self.inplace, self.beta = inplace, beta
    super().__init__()
  def forward(self, x: Tensor) -> Tensor: return x.swish(self.beta)
  def __repr__(self) -> str: return f"LeakyReLU(beta={self.beta} inplace={self.inplace})"
  def inner_repr(self) -> str: return f"inplace={self.inplace}"

class Softplus(Module):
  def __init__(self, inplace: Tensor = False):
    self.inplace = inplace
    super().__init__()
  def forward(self, x: Tensor) -> Tensor: return x.softplus()
  def __repr__(self) -> str: return f"Softplus(inplace={self.inplace})"
  def inner_repr(self) -> str: return f"inplace={self.inplace}"