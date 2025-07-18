from .parameter import Parameter
from abc import ABC, abstractmethod
from collections import OrderedDict
import inspect, pickle, os
from ..tensor import Tensor

class Module(ABC):
  def __init__(self): self._modules, self._params, self._grads = OrderedDict(), OrderedDict(), OrderedDict()
  @abstractmethod
  def forward(self, *args, **kwargs): raise NotImplementedError("Forward function not implemented yet!")
  def __call__(self, *args, **kwds): return self.forward(*args, **kwds)
  def train(self):
    for _, _, param in self.parameters(): param.requires_grad = True
  def eval(self):
    for _, _, param in self.parameters(): param.requires_grad = False
  def parameters(self):
    for name, value in inspect.getmembers(self):
      if isinstance(value, Parameter): yield self, name, value
      elif isinstance(value, Module):
        for module, param_name, param in value.parameters(): yield module, param_name, param

  def modules(self): yield from self._modules.values()
  def gradients(self):
    for module in self.modules(): yield module._grads

  def zero_grad(self): any(parameter.zero_grad() for _, _, parameter in self.parameters())
  def get_name(self): return self.__class__.__name__

  def __repr__(self):
    string, tab, modules = f"{self.get_name()}(", "   ", self._modules
    if modules == {}: string += f"\n{tab}(parameters): {self.inner_repr()}"
    else:
      for key, module in modules.items(): string += f"\n{tab}({key}): {module.get_name()}({module.inner_repr()})"
    return f"{string}\n)"

  def n_params(self):
    total = 0
    for _, _, param in self.parameters(): total += param.size
    return total

  def save(self, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state_dict = {}     # Create state dict
    for module, param_name, param in self.parameters():
      # Create a hierarchical key like "layer1.weight", "layer1.bias", etc.
      if module is self: key = param_name
      else:
        module_name = None
        for name, mod in self._modules.items():
          if mod is module:
            module_name = name
            break
        if module_name: key = f"{module_name}.{param_name}"
        else: key = param_name
      state_dict[key] = param.tolist()    
    with open(filepath, 'wb') as f: pickle.dump(state_dict, f)
    print(f"Model saved to {filepath}")

  def load(self, filepath):
    with open(filepath, 'rb') as f: state_dict = pickle.load(f)
    param_map = {}
    for module, param_name, param in self.parameters():
      if module is self: key = param_name
      else:
        # Find the module name by searching through _modules
        module_name = None
        for name, mod in self._modules.items():
          if mod is module:
            module_name = name
            break

        if module_name: key = f"{module_name}.{param_name}"
        else: key = param_name
        param_map[key] = param

    for key, data in state_dict.items():
      if key in param_map:
        # Create new tensor with the loaded data
        new_tensor = Tensor(data, dtype=param_map[key].dtype)
        # Copy the data to the parameter
        param_map[key].data = new_tensor.data
        param_map[key].shape = new_tensor.shape
        param_map[key].size = new_tensor.size
        param_map[key].ndim = new_tensor.ndim
        param_map[key].strides = new_tensor.strides
      else: print(f"Warning: Parameter '{key}' not found in model")
    print(f"Model loaded from {filepath}")