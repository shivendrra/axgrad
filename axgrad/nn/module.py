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

  def _build_module_path_map(self):
    module_to_path = {self: ""}        
    def traverse(current_module, current_path):
      for attr_name in dir(current_module):
        if attr_name.startswith('_'): continue
        try:
          attr_value = getattr(current_module, attr_name)
          if isinstance(attr_value, Module) and attr_value is not current_module:
            new_path = f"{current_path}.{attr_name}" if current_path else attr_name
            module_to_path[attr_value] = new_path
            traverse(attr_value, new_path)     # recursively traverse submodules
        except: continue
    traverse(self, "")
    return module_to_path

  def save(self, filepath:str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    module_to_path, state_dict = self._build_module_path_map(), {}
    for module, param_name, param in self.parameters():
      module_path = module_to_path.get(module, "")
      if module_path: key = f"{module_path}.{param_name}"
      else: key = param_name
      state_dict[key] = param.tolist()
    with open(filepath, 'wb') as f: pickle.dump(state_dict, f)
    print(f"Saved {len(state_dict)} parameters to {filepath}")

  def load(self, filepath: str):
    with open(filepath, 'rb') as f: state_dict = pickle.load(f)        
    module_to_path = self._build_module_path_map()
    param_map = {} # creating parameter mapping with proper hierarchical keys
    for module, param_name, param in self.parameters():
      module_path = module_to_path.get(module, "")
      if module_path: key = f"{module_path}.{param_name}"
      else: key = param_name
      param_map[key] = param

    loaded_count = 0
    for key, data in state_dict.items():
      if key in param_map:
        param = param_map[key]
        new_tensor = Tensor(data, dtype=param.dtype)
        param.data = new_tensor.data
        param.shape = new_tensor.shape
        param.size = new_tensor.size
        param.ndim = new_tensor.ndim
        param.strides = new_tensor.strides
        loaded_count += 1
      else: print(f"  WARNING: {key} not found in current model")
    print(f"Successfully loaded {loaded_count}/{len(state_dict)} parameters")

  def named_parameters(self):
    """ Alternative method that yields (name, parameter) tuples with proper hierarchical naming """
    module_to_path = self._build_module_path_map()        
    for module, param_name, param in self.parameters():
      module_path = module_to_path.get(module, "")
      if module_path: full_name = f"{module_path}.{param_name}"
      else: full_name = param_name
      yield full_name, param

  def state_dict(self):
    state = {}
    for name, param in self.named_parameters(): state[name] = param.tolist()
    return state

  def load_state_dict(self, state_dict):
    param_map = {}
    for name, param in self.named_parameters(): param_map[name] = param
    loaded_count = 0
    for name, data in state_dict.items():
      if name in param_map:
        param = param_map[name]
        new_tensor = Tensor(data, dtype=param.dtype)
        param.data = new_tensor.data
        param.shape = new_tensor.shape
        param.size = new_tensor.size
        param.ndim = new_tensor.ndim
        param.strides = new_tensor.strides
        loaded_count += 1
      else: print(f"  WARNING: {name} not found in current model")
    print(f"Successfully loaded {loaded_count}/{len(state_dict)} parameters")