from ..engine import Value

def absolute(val: list):
  return [abs(v) for v in val]

class Loss:

  def error(self, trg, prd):
    return prd - trg

  def mse_loss(self, trg, prd):
    loss = (sum(self.error(ygt, yout)**2 for ygt, yout in zip(trg, prd))) / len(trg)
    return loss

  def mae_loss(self, trg, prd):
    loss = Value(sum(absolute(self.error(ygt, yout).data) for ygt, yout in zip(trg, prd))) / len(trg)
    return loss
  
  def hinge_loss(self, trg, prd):
    loss = (sum(max(0 , 1-trg*prd)))
    return loss
  
  def smooth_mae(self, trg, prd, s_factor=1):
    pass