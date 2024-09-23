from collections import OrderedDict
from .parameters import Parameter
import pickle

class Module:
  def __init__(self) -> None:
    super().__init__()
    self._modules = OrderedDict()
    self._params = OrderedDict()
    self._grads = OrderedDict()
    self.training = True
  
  def forward(self, *inputs, **kwargs):
    raise NotImplementedError("forward not written")
  
  def __call__(self, *inputs, **kwargs):
    return self.forward(*inputs, **kwargs)
  
  def train(self):
    self.training = True
    for param in self.parameters():
      param.requires_grad = True
  
  def eval(self):
    self.training = True
    for param in self.parameters():
      param.requires_grad = False
  
  def modules(self):
    yield from self._modules.values()

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
      module_str += '  (' + key + '): Parameters: ' + str(param.tolist()) + '\n'
    module_str += ')'
    return module_str
  
  def n_param(self):
    total = 0
    for param in self.parameters():
      total += param.numel()
    return total

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