from collections import OrderedDict

class Module:
  def __init__(self) -> None:
    self._modules = OrderedDict()
    self._params = OrderedDict()
    self._grads = OrderedDict()
    self.trainig = True
  
  def forward(self):
    raise NotImplementedError('forward not written')
  
  def train(self):
    self.trainig = True
    for param in self.parameters():
      param.require_grad = True
  
  def eval(self):
    self.trainig = False
    for param in self.parameters():
      param.require_grad = False
  
  def modules(self):
    raise NotImplementedError('train not written')
  
  def train():
    raise NotImplementedError('train not written')
  
  def zero_grad(self):
    for _, _, parameter in self.parameters():
      parameter.zero_grad()