from ..tensor import Tensor
from .._core import DType, lib
from ..helpers import ShapeHelp, DtypeHelp

from ctypes import c_int, c_size_t, c_float

class Parameter(Tensor):
  def __init__(self, shape: tuple, dtype: str = "float32") -> None:
    s, sz, nd, sa = ShapeHelp.process_shape(shape)
    data = lib.randn_tensor(sa, c_size_t(sz), c_size_t(nd), c_int(DtypeHelp._parse_dtype(dtype))).contents
    super().__init__(data, dtype, requires_grad=True)
    self.shape, self.size, self.ndim, self.strides = tuple(s), sz, nd, ShapeHelp.get_strides(s)
    self._is_parameter:bool = True; self._name:str = None

  @property
  def is_parameter(self) -> bool: return self._is_parameter
  def set_name(self, name: str) -> None: self._name = name
  def get_name(self) -> str: return self._name or "Parameter"
  
  def zero_grad(self) -> None:
    if self.grad is not None:
      zeros_data = lib.zeros_tensor((c_int * self.ndim)(*self.shape), c_size_t(self.size), c_size_t(self.ndim), c_int(DtypeHelp._parse_dtype(self.dtype))).contents
      self.grad = Tensor(zeros_data, self.dtype, requires_grad=False)
      self.grad.shape, self.grad.size, self.grad.ndim, self.grad.strides = self.shape, self.size, self.ndim, self.strides
  
  def __repr__(self) -> str:
    name_str = f"({self.get_name()})" if self._name else ""
    return f"Parameter{name_str} containing:\n{super().__str__()}"
  
  def clone(self) -> "Parameter":
    data_list = self.tolist()
    new_param = Parameter.__new__(Parameter)
    new_param.__init__(self.shape, self.dtype, self.requires_grad)

    if isinstance(data_list, list): flat_data = ShapeHelp.flatten(data_list)
    else: flat_data = [data_list]
    data_ctypes, shape_ctypes = (c_float * self.size)(*flat_data), (c_int * self.ndim)(*self.shape)
    new_param.data = lib.create_tensor(data_ctypes, c_size_t(self.ndim), shape_ctypes, c_size_t(self.size), c_int(DtypeHelp._parse_dtype(self.dtype))).contents
    new_param.shape, self.size, self.ndim, self.strides = self.shape, self.size, self.ndim, self.strides
    new_param._name, new_param.requires_grad = self._name, self.requires_grad
    return new_param

  def detach(self) -> Tensor:
    detached = Tensor(self.data, self.dtype, requires_grad=False)
    detached.shape, detached.size, detached.ndim, detached.strides = self.shape, self.size, self.ndim, self.strides
    return detached