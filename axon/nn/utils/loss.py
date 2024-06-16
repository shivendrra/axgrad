from ...tensor import tensor

class MSE:
  def __init__(self, target:tensor, predicted:tensor):
    self.target = target
    self.predicted = predicted
  
  def mse(self, trg, prd):
    diff = trg - prd
    sq = diff ** 2
    loss = sq / len(prd.data)
    return loss