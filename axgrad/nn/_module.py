"""
  @nn/_module.py main module file for all nn blocks
  @brief main parent class for manipulating, storing all the parameters & nn building classes/functions
  @comments:
  - works like torch.nn.Module
  - can be used to inherit to build new nn blocks
"""

from collections import OrderedDict
from ._parameters import Parameter
import pickle

class Module:
  def __init__(self) -> None: super().__init__(); self._modules, self._params, self._grads, self.training = OrderedDict(), OrderedDict(), OrderedDict(), True
  def forward(self, *inputs, **kwargs): raise NotImplementedError("forward not written")  
  def __call__(self, *inputs, **kwargs): return self.forward(*inputs, **kwargs)
  def modules(self): yield from self._modules.values()
  def n_param(self): return sum(param.numel() for param in self.parameters())

  def parameters(self):
    params = []
    for param in self._params.values():
      params.append(param)
    for module in self._modules.values():
      params.extend(module.parameters())
    return params
  
  def train(self):
    self.training_mode = True

  def eval(self):
    self.training_mode = False

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

  def display_grads(self):
    """ Display gradients for all parameters in the module. """
    print(f"Displaying gradients for module: {self.__class__.__name__}")
    for name, param in self._params.items():
      grad = param.grad if param.grad is not None else "None"
      print(f"Parameter: {name} -> Gradient: {grad}")
      
    for module_name, module in self._modules.items():
      print(f"\nSubmodule: {module_name}")
      module.display_grads()