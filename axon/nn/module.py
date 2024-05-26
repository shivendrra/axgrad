from collections import OrderedDict
from .parameter import Parameter
import pickle

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
    for param in self.parameters():
      param.zero_grad()
  
  def parameters(self):
    params = []
    for param in self._params.values():
      params.append(param)
    for module in self._modules.values():
      params.extend(module.parameters())
    return params
  
  def save_dict(self):
    state = OrderedDict()
    for name, param in self._params.items():
      state[name] = param.tolist()
    for name, module in self._modules.items():
      state[name] = module.save_dict()
    return state
  
  def save(self, filename='model.pickle'):
    with open(filename, 'wb') as f:
      pickle.dump(self.save_dict(), f)
  
  def load(self, filename='model.pickle'):
    with open(filename, 'rb') as f:
      state = pickle.load(f)
    self.load_dict(state)
  
  def load_dict(self, state):
    for name, value in state.items():
      if isinstance(value, dict):
        self._modules[name].load_dict(value)
      else:
        self._params[name].data = value
  
  def __setattr__(self, key, value):
    if isinstance(value, Module):
      self._modules[key] = value
    elif isinstance(value, Parameter):
      self._params[key] = value
    super().__setattr__(key, value)
  
  def __repr__(self):
    module_str = self.__class__.__name__ + '(\n'
    for key, module in self._modules.items():
      module_str += '  (' + key + '): ' + repr(module) + '\n'
    for key, param in self._params.items():
      module_str += '  (' + key + '): Parameter containing: ' + str(param.tolist()) + '\n'
    module_str += ')'
    return module_str