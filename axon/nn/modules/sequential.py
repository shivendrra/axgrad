from ..module import Module

class Sequential(Module):
  def __init__(self, *layers):
    super().__init__()
    self.layers = layers
  
  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def __call__(self, x):
    return self.forward(x)
  
  def parameters(self):
    params = []
    for layer in self.layers:
      if hasattr(layer, 'parameters'):
        params.extend(layer.parameters())
    return params

  def __repr__(self):
    layer_names = [type(layer).__name__ for layer in self.layers]
    return f"<SequentialLayer({','.join(layer_names)})>"