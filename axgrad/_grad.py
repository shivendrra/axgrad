"""
  @file _grad.py
  @brief contains grad class for maintaining grads
    * grad() inherits all the neccessary functionalities from the _tensor base class
    * only extends the __str__ pretty printing capabilities for now!
"""
from ._core import _tensor

class grads(_tensor):
  def __init__(self, data, dtype = None):
    super().__init__(data, dtype)
  def __repr__(self) -> str:
    return f"tensor.grad({self.data})"
  def __str__(self) -> str:
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