import math

class Loss:

  def error(self, trg, prd):
    return (prd - trg)

  def mean_square_error(self, trg, prd):
    loss = (sum(self.error(ygt, yout)**2 for ygt, yout in zip(trg, prd))) / len(trg)
    return loss

  def mean_absolute_error(self, trg, prd):
    loss = (sum(self.error(ygt, yout) for ygt, yout in zip(trg, prd))) / len(trg)
    return loss