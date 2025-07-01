from ..tensor import Tensor
from ..nn.parameter import Parameter
from ..nn.module import Module
from .._core import lib
from ..helpers import ShapeHelp, DtypeHelp
from ctypes import c_int, c_size_t, c_float
from typing import Union, List, Iterator, Tuple

class SGD:
  def __init__(self, parameters: Union[Iterator[Tuple[Module, str, Parameter]], List[Parameter]], lr: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0, dampening: float = 0.0, nesterov: bool = False):
    if nesterov and (momentum <= 0 or dampening != 0):
      raise ValueError("Nesterov momentum requires a momentum and zero dampening")
  
    self.lr, self.momentum, self.weight_decay = lr, momentum, weight_decay
    self.dampening, self.nesterov = dampening, nesterov
    self.param_groups = []
    
    # handling both parameter iterators from Module.parameters() and direct parameter lists
    if hasattr(parameters, '__iter__') and not isinstance(parameters, list):
      # iterator from Module.parameters() - yields (module, name, param)
      params = []
      for module, name, param in parameters:
        if isinstance(param, Parameter): params.append(param)
      self.param_groups.append({'params': params})
    else:
      # direct list of parameters
      params = [p for p in parameters if isinstance(p, Parameter)]
      self.param_groups.append({'params': params})

    # initialize momentum buffers
    self.state = {}
    for group in self.param_groups:
      for param in group['params']:
        if param not in self.state: self.state[param] = {}

  def zero_grad(self):
    for group in self.param_groups:
      for param in group['params']: param.zero_grad()

  def step(self):
    for group in self.param_groups:
      for param in group['params']:
        if param.grad is None: continue
        grad = param.grad
        param_state = self.state[param]
        if self.weight_decay != 0: grad = grad + param * self.weight_decay     # applying weight decay
        # Apply momentum
        if self.momentum != 0:
          if 'momentum_buffer' not in param_state:
            # Initialize momentum buffer with zeros
            zeros_data = lib.zeros_tensor((c_int * param.ndim)(*param.shape), c_size_t(param.size), c_size_t(param.ndim), c_int(DtypeHelp._parse_dtype(param.dtype))).contents
            buf = Tensor(zeros_data, param.dtype, requires_grad=False)
            buf.shape, buf.size, buf.ndim, buf.strides = param.shape, param.size, param.ndim, param.strides
            param_state['momentum_buffer'] = buf

          buf = param_state['momentum_buffer']
          buf = buf * self.momentum + grad * (1 - self.dampening)
          param_state['momentum_buffer'] = buf

          if self.nesterov: grad = grad + buf * self.momentum
          else: grad = buf

        # update parameters: param = param - lr * grad
        update = grad * self.lr
        param.data = lib.sub_tensor(param.data, update.data).contents

  def add_param_group(self, param_group: dict):
    assert isinstance(param_group, dict), "param_group must be a dict"
    params = param_group['params']
    if isinstance(params, Parameter): params = [params]
    elif not isinstance(params, list): params = list(params)

    param_group['params'] = params
    self.param_groups.append(param_group)
    # Initialize state for new parameters
    for param in params:
      if param not in self.state: self.state[param] = {}

  def __repr__(self) -> str: return (f"SGD(lr={self.lr}, momentum={self.momentum}, weight_decay={self.weight_decay}, " f"dampening={self.dampening}, nesterov={self.nesterov})")
  def get_lr(self) -> float: return self.lr
  def set_lr(self, lr: float): self.lr = lr