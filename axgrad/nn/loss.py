from ..tensor import Tensor
from ..autograd.functions import *
from ctypes import c_int, c_size_t, c_float
import math

class MSELoss:
  def __init__(self, reduction: str = "mean"):
    assert reduction in ["mean", "sum", "none"], f"Reduction '{reduction}' not supported. Use 'mean', 'sum', or 'none'"
    self.reduction = reduction

  def forward(self, pred: Tensor, target: Tensor) -> Tensor:
    assert pred.shape == target.shape, f"Prediction shape {pred.shape} doesn't match target shape {target.shape}"
    # Calculate (pred - target)^2
    diff = pred - target
    squared_diff = diff ** 2
    if self.reduction == "none": return squared_diff
    elif self.reduction == "sum": return squared_diff.sum()
    else: return squared_diff.sum() * (1.0 / target.size)  # mean

  def __call__(self, pred: Tensor, target: Tensor) -> Tensor: return self.forward(pred, target)
  def __repr__(self) -> str: return f"MSELoss(reduction='{self.reduction}')"

class MAELoss:
  def __init__(self, reduction: str = "mean"):
    assert reduction in ["mean", "sum", "none"], f"Reduction '{reduction}' not supported. Use 'mean', 'sum', or 'none'"
    self.reduction = reduction

  def forward(self, pred: Tensor, target: Tensor) -> Tensor:
    assert pred.shape == target.shape, f"Prediction shape {pred.shape} doesn't match target shape {target.shape}"
    # Calculate |pred - target|
    diff = pred - target
    abs_diff = diff.abs()
    if self.reduction == "none": return abs_diff
    elif self.reduction == "sum": return abs_diff.sum()
    else: return abs_diff.sum() * (1.0 / target.size)  # mean

  def __call__(self, pred: Tensor, target: Tensor) -> Tensor: return self.forward(pred, target)
  def __repr__(self) -> str: return f"MAELoss(reduction='{self.reduction}')"

class CrossEntropy:
  def __init__(self, reduction: str = "sum"):
    assert reduction in ["mean", "sum", "none"], f"Reduction '{reduction}' not supported. Use 'mean', 'sum', or 'none'"
    self.reduction = reduction

  def forward(self, pred: Tensor, target: Tensor) -> Tensor:
    max_logits = pred.max(axis=1, keepdims=True)
    log_probs = pred - max_logits - ((pred - max_logits).exp()).sum(axis=1, keepdims=True).log()
    N = pred.shape[0]
    loss = -log_probs[N, target]
    if self.reduction == "none": return loss
    elif self.reduction == "sum": return loss.sum()
    else: return loss.mean()

  def __call__(self, pred: Tensor, target: Tensor) -> Tensor: return self.forward(pred, target)
  def __repr__(self) -> str: return f"CrossEntropy(reduction={self.reduction})"