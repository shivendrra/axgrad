"""
  @nn/_module.py main module file for all nn blocks
  @brief main parent class for manipulating, storing all the parameters & nn building classes/functions
  @comments:
  - works like torch.nn.Module
  - can be used to inherit to build new nn blocks
"""

from collections import OrderedDict
from ._parameters import Parameter

class Module:
  def __init__(self) -> None: super().__init__(); self._modules, self._params, self._grads, self.training = OrderedDict(), OrderedDict(), OrderedDict(), True
  def forward(self, *inputs, **kwargs): raise NotImplementedError("forward not written")  
  def __call__(self, *inputs, **kwargs): return self.forward(*inputs, **kwargs)
  def modules(self): yield from self._modules.values()
  def n_param(self): return sum(param.numel() for param in self.parameters())

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