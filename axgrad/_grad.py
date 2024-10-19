"""
  @_grad.py main grad file
  @brief contains grad class for maintaining grads
"""
from .helpers.utils import _zeros
from typing import Any, Iterator, List, Union

class grads:
  def __init__(self, data=None, shape=None) -> None: self.data = data if data is not None else _zeros(shape=shape) if shape is not None else _zeros(shape=(1,1))
  def __repr__(self):
    return f"tensor.grad({self.data})"
  def __str__(self):
    def format_element(element): return [format_element(sub_element) for sub_element in element] if isinstance(element, list) else f"{element:.4f}"
    formatted_data = format_element(self.data)
    def truncate_list(data, max_items=8): return data[:max_items//2] + ["..."] + data[-max_items//2:] if len(data) > max_items else data
    def format_data(data, level=0):
      if isinstance(data[0], list):
        if len(data) > 8:
          data = truncate_list(data)  # truncate rows if there are more than 8 arrays
        inner = ",\n".join(["  " * (level + 1) + format_data(sub_data, level + 1) for sub_data in data])
        return f"[\n{inner}\n" + "  " * level + "]"
      else:
        # truncate individual row elements if they exceed 8
        data = truncate_list(data)
        return "[" + ", ".join(data) + "]"
    formatted_str = format_data(formatted_data, 0)
    formatted_str = formatted_str.replace("\t", " ")
    return f"grad({formatted_str})"
  
  def __getitem__(self, index:tuple):
    if isinstance(index, tuple):
      data = self.data
      for idx in index[:-1]:
        data = data[idx]
      return data[index[-1]]
    else: return self.data[index]

  def __setattr__(self, name: str, value: Any) -> None:
    super().__setattr__(name, value)
  
  def __setitem__(self, index:tuple, value: Any) -> None:
    if isinstance(index, tuple):
      data = self.data
      for idx in index[:-1]:
        data = data[idx]
      data[index[-1]] = value
    else: self.data[index] = value

  def __iter__(self) -> Iterator: yield from self.data

  def __mul__(self, scalar):
    other = other if isinstance(other, grads) else grads(data=other, shape=None)
    def _ops(data):
      if isinstance(data, list):
        return [_ops(_d) for _d in data]
      return data * scalar
    return grads(data=_ops(self.data), shape=None)

  def __sub__(self, other):
    other = other if isinstance(other, grads) else grads(data=other, shape=None)
    def _ops(data, other_data):
      if isinstance(data, list):
        return [_ops(_d, _od) for _d, _od in zip(data, other_data)]
      return data - other_data
    return grads(data=_ops(self.data, other.data), shape=None)
  
  def __add__(self, other):
    other = other if isinstance(other, grads) else grads(data=other, shape=None)
    def _ops(data, other_data):
      if isinstance(data, list):
        return [_ops(_d, _od) for _d, _od in zip(data, other_data)]
      return data + other_data
    return grads(data=_ops(self.data, other.data), shape=None)