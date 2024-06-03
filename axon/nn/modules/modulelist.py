from ..module import Module

class ModuleList(Module):
  def __init__(self, *modules):
    super().__init__()
    self.module = list(modules)

  def append(self, modules):
    self.module.append(modules)
  
  def extend(self, modules):
    self.module.extend(modules)

  def __getitem__(self, idx):
    return self.module[idx]
  
  def __len__(self):
    return len(self.module)
  
  def __iter__(self):
    return iter(self.module)

  def forward(self, x):
    for module in self.module:
      x = module(x)
    return x
  
  def __call__(self, x):
    return self.forward(x)
  
  def parameters(self):
    params = []
    for module in self.module:
      if hasattr(module, 'parameters'):
        params.extend(module.parameters())
    return params

  def __repr__(self):
    module_names = [type(layer).__name__ for layer in self.module]
    return f"<ModuleListLayer({','.join(module_names)})>"