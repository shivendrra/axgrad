from ..engine import Value

def absolute(val: list):
  return [abs(v) for v in val]

class Loss:

  def error(self, trg, prd):
    """
      simple difference function

      Args:
        trg (int or floar): ground truth
        prd (int or float): logits

      Returns:
        int or float: difference b/w logits & ground truth
    """
    return prd - trg

  def mse_loss(self, trg, prd):
    """
      simple mean squared error loss function
        
        'mse_loss = sum(error**2)/len(trg)'

      Args:
        trg (list): list containing target tokens
        prd (list): list containing logits

      Returns:
        axgrad.Value: float value of loss as Value object for backprop
    """
    loss = sum(self.error(ygt, yout)**2 for ygt, yout in zip(trg, prd)) / len(trg)
    return loss if isinstance(loss, Value) else Value(loss)

  def mae_loss(self, trg, prd):
    """
      simple mean absolute error loss function
        
        'mae_loss = sum(mod(error))/len(trg)'

      Args:
        trg (list): list containing target tokens
        prd (list): list containing logits

      Returns:
        axgrad.Value: float value of loss as Value object for backprop
    """
    loss = (sum(absolute(self.error(ygt, yout).data) for ygt, yout in zip(trg, prd))) / len(trg)
    return loss if isinstance(loss, Value) else Value(loss)
  
  def hinge_loss(self, trg, prd):
    """
      hinge loss function

        'loss = max(0, 1 - prd * trg)'

      Args:
        trg (list): list containing target tokens
        prd (list): list containing logits

      Returns:
        axgrad.Value: float value of loss as Value object for backprop
    """
    loss = (sum(max(0 , 1-trg*prd)))
    return loss