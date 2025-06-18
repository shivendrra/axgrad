"""
  @tensor.py Main tensor class
  @breif Code contains axgrad.tensor class to perform backprop
  @comments
  - conjusted to save total lines of code
  - has basic functions & operations with backward function in same file
  - entrypoint to whole axgrad.tensor class & functions
"""

from typing import *
from copy import deepcopy

from .helpers.shape import *
from .helpers.utils import _zeros, _ones
from ._dtype import *
from .ops.functionals import *
from .helpers.utils import _ones
from .autograd._backward import Backward
from ._grad import grads
from ._core import _tensor

int8, int16, int32, int64, long = "int8", "int16", "int32", "int64", "long"
float16, float32, float64, double = "float16", "float32", "float64", "double"

class tensor(_tensor):
  training_mode = True
  
  def __init__(self, data, requires_grad:bool=True, dtype:Optional[Literal["int8", "int16", "int32", "int64", "float16", "float32", "float64", "long", "double"]]=None) -> None:
    super().__init__(data, dtype)
    self.prev = set()
    self._backward = lambda: None

    # only if requires_grad is true
    if requires_grad is True:
      if self.is_scalar:
        self.grad = grads(data=[0.0], dtype=self.dtype)
      else:
        self.grad = grads(data=_ones(self.shape), dtype=self.dtype)
      self.requires_grad, self.grad_fn = requires_grad, "<NotSet>"
    else:
      self.grad, self.grad_fn, self.requires_grad = None, None, False

  def __repr__(self) -> str:
    # basic representation for easy operation computing & no element or floating point truncation
    return f"tensor([{self.data}])"

  def __str__(self) -> str:
    # formated version computing for prettier outputs
    # truncates some of the elements & sub tensors if more than 8
    # displays additional information

    def format_element(element):
      if isinstance(element, list):
        return [format_element(sub_element) for sub_element in element]
      if self.dtype == int8 or self.dtype == int16 or self.dtype == int32 or self.dtype == int64 or self.dtype == long:
        return f"{element:.0f}."
      if self.dtype == float16:
        return f"{element:.2f}"
      if self.dtype == float32:
        return f"{element:.3f}"
      return f"{element:.4f}"
    
    formatted_data = format_element(self.data)
    
    def truncate_list(data, max_items=8): 
      return data[:max_items//2] + ["..."] + data[-max_items//2:] if len(data) > max_items else data
    
    def format_data(data, level=0):
      if isinstance(data[0], list):
        if len(data) > 8:
          data = truncate_list(data)  # Truncate rows if there are more than 8 arrays
        inner = ",\n".join(["  " * (level + 1) + format_data(sub_data, level + 1) for sub_data in data])
        return f"[\n{inner}\n" + "  " * level + "]"
      else:
        # Truncate individual row elements if they exceed 8
        data = truncate_list(data)
        return "[" + ", ".join(data) + "]"
    
    formatted_str = format_data(formatted_data, 0)
    formatted_str = formatted_str.replace("\t", " ")
    return f"tensor({formatted_str}, dtype={self.dtype}, grad_fn={self.grad_fn})\n"  if self.requires_grad else f"tensor({formatted_str}, dtype={self.dtype})\n"
  
  # property attributes --------
  
  @property
  def ndim(self) -> int:
    return len(self.shape)
  
  @property
  def numel(self) -> int:
    out = 1
    for dim in self.shape:
      out *= dim
    return out

  def training(self, mode: bool = True) -> None:
    """
    sets the training mode of the tensor. when in training mode, gradients are tracked.
    args:
      mode (bool): True for training mode, False for inference mode.
    """
    tensor.training_mode = mode  # toggle the global training mode
    self.requires_grad = mode 

  def evalutation(self) -> None:
    """switches to evaluation mode (no gradient tracking)."""
    self.training(mode=False)

  def astype(self, dtype:Optional[Literal["int8", "int16", "int32", "int64", "float16", "float32", "float64"]]) -> "tensor":
    new_data = Dtype.handle_conversion(self.data, dtype)
    out = tensor(new_data, self.requires_grad, dtype)
    out.prev, out.grad, out.grad_fn = self.prev, self.grad, self.grad_fn
    return out

  def view(self, *new_shape:Union[int, list, tuple]) -> "tensor":
    if isinstance(new_shape[0], list) or isinstance(new_shape[0], tuple):
      new_shape = tuple(new_shape[0])
    elif isinstance(new_shape[0], int):
      new_shape = tuple(new_shape)
    self.make_contiguous()
    flat_data = self.flatten()
    total_elements = len(flat_data)
    if total_elements != self.numel:
      raise ValueError("Total elements in new shape must match the number of elements in the original tensor")
    out = tensor(reshape(self.data, new_shape), requires_grad=self.requires_grad, dtype=self.dtype)
    out.prev, out.grad_fn, out._backward = (self, ), "<ViewBackwards>", Backward.view_backwards(self, out, self.shape)
    return out

  def backward(self, gradient: Optional["tensor"] = None, retain_graph: bool = False) -> None:
    """
    computes gradients of current tensor w.r.t. graph leaves using backpropagation.
    
    args:
      gradient (tensor, optional): gradient w.r.t. the tensor. if None, tensor must be scalar.
      retain_graph (bool): if False, the graph used to compute grads will be freed.
    """
    if not self.requires_grad:
      raise RuntimeError("tensor does not require gradients, cannot compute backward pass")
    
    # for scalar tensors, gradient defaults to 1.0
    if gradient is None:
      if not self.is_scalar:
        raise RuntimeError("grad can be implicitly created only for scalar outputs")
      gradient = tensor([1.0], requires_grad=False, dtype=self.dtype)

    # ensure gradient has same shape as this tensor
    if gradient.shape != self.shape:
      raise RuntimeError(f"gradient shape {gradient.shape} doesn't match tensor shape {self.shape}")

    # topological sort using kahn's algorithm
    topo_order, visited, in_degree = [], set(), {}
    
    def build_graph(node):
      """build computational graph and calculate in-degrees"""
      if node in visited:
        return
      visited.add(node)
      
      if node not in in_degree:
        in_degree[node] = 0
      
      if hasattr(node, 'prev'):
        for parent in node.prev:
          if parent.requires_grad:
            build_graph(parent)
            if parent not in in_degree:
              in_degree[parent] = 0
            in_degree[parent] += 1
    
    # build the computational graph starting from this tensor
    build_graph(self)

    # kahn's algorithm for topological sorting
    queue = [node for node in in_degree if in_degree[node] == 0]

    while queue:
      current = queue.pop(0)
      topo_order.append(current)
      
      if hasattr(current, 'prev'):
        for parent in current.prev:
          if parent.requires_grad and parent in in_degree:
            in_degree[parent] -= 1
            if in_degree[parent] == 0:
              queue.append(parent)

    # initialize gradients - this tensor gets the input gradient
    if self.grad is None:
      if self.is_scalar:
        self.grad = grads(data=[0.0], dtype=self.dtype)
      else:
        self.grad = grads(data=_zeros(self.shape), dtype=self.dtype)
    
    # accumulate gradient for this tensor
    self._accumulate_grad(gradient)
    
    # propagate gradients backward through the computational graph
    for node in reversed(topo_order):
      if node != self and node.requires_grad and hasattr(node, '_backward'):
        try:
          node._backward()
        except Exception as e:
          raise RuntimeError(f"error during backward pass at {node.grad_fn}: {str(e)}")
    
    # optionally free the computational graph
    if not retain_graph:
      self._free_graph()
  
  def _accumulate_grad(self, grad: "tensor") -> None:
    """accumulate gradients, handling the case where grad might be None"""
    if self.grad is None:
      if self.is_scalar:
        self.grad = grads(data=[0.0], dtype=self.dtype)
      else:
        self.grad = grads(data=_zeros(self.shape), dtype=self.dtype)
    
    # element-wise addition of gradients
    if grad.is_scalar and self.is_scalar:
      self.grad.data[0] += grad.data[0]
    else:
      self._add_gradients_elementwise(self.grad.data, grad.data)
  
  def _add_gradients_elementwise(self, grad_data, new_grad_data):
    """recursively add gradients element-wise"""
    if isinstance(grad_data, list) and isinstance(new_grad_data, list):
      for i in range(len(grad_data)):
        if isinstance(grad_data[i], list):
          self._add_gradients_elementwise(grad_data[i], new_grad_data[i])
        else:
          grad_data[i] += new_grad_data[i]
    else:
      # handle scalar case
      grad_data += new_grad_data
  
  def _free_graph(self) -> None:
    """free the computational graph to save memory"""
    visited = set()
    
    def free_node(node):
      if node in visited:
        return
      visited.add(node)
      
      if hasattr(node, 'prev'):
        for parent in node.prev:
          free_node(parent)
        node.prev = set()
      
      if hasattr(node, '_backward'):
        node._backward = lambda: None
    
    free_node(self)
  
  def zero_grad(self) -> None:
    """zeros out the gradients of this tensor"""
    if self.grad is not None:
      if self.is_scalar:
        self.grad.data = [0.0]
      else:
        self.grad.data = _zeros(self.shape)
  
  def detach(self) -> "tensor":
    """returns a new tensor detached from the computational graph"""
    out = tensor(deepcopy(self.data), requires_grad=False, dtype=self.dtype)
    return out