from .parameter import Parameter
from abc import ABC, abstractmethod
from collections import OrderedDict
import inspect

class Module(ABC):
  def __init__(self): self._modules, self._params, self._grads = OrderedDict(), OrderedDict(), OrderedDict()
  @abstractmethod
  def forward(self, *args, **kwargs): raise NotImplementedError("Forward function not implemented yet!")
  def __call__(self, *args, **kwds): return self.forward(*args, **kwds)
  def train(self):
    for param in self.parameters(): param.requires_grad = True
  def eval(self):
    for param in self.parameters(): param.requires_grad = False
  def parameters(self):
    for name, value in inspect.getmembers(self):
      if isinstance(value, Parameter):
        yield self, name, value
      elif isinstance(value, Module):
        yield from value.parameters()

  def modules(self): yield from self._modules.values()
  def gradients(self):
    for module in self.modules():
      yield module._grads

  def zero_grad(self):
    for _, _, parameter in self.parameters():
      parameter.zero_grad()

  def get_name(self):
    return self.__class__.__name__

  def __repr__(self):
    string = f"{self.get_name()}("
    tab = "   "
    modules = self._modules
    if modules == {}: string += f"\n{tab}(parameters): {self.inner_repr()}"
    else:
      for key, module in modules.items():
        string += f"\n{tab}({key}): {module.get_name()}({module.inner_repr()})"
    return f"{string}\n)"