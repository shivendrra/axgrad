import ctypes, os, sys, platform, sysconfig
from ctypes import Structure, c_float, c_double, c_int, c_int8, c_int16, c_int32, c_int64, c_uint8, c_uint16, c_uint32, c_uint64, c_size_t, c_void_p, c_char_p, POINTER
from typing import *

def _get_lib_path():
  pkg_dir = os.path.dirname(__file__)
  possible_names = ['tensor', 'libtensor']
  possible_exts = ['.pyd', '.dll', '.so', '.dylib', sysconfig.get_config_var('EXT_SUFFIX') or '']
  search_dirs = [pkg_dir, os.path.join(pkg_dir, 'lib'), os.path.join(pkg_dir, '..', 'build')]
  
  for search_dir in search_dirs:
    if not os.path.exists(search_dir): continue
    for root, dirs, files in os.walk(search_dir):
      for file in files:
        for name in possible_names:
          if file.startswith(name) and any(file.endswith(ext) for ext in possible_exts if ext):
            return os.path.join(root, file)
  
  raise FileNotFoundError(f"Could not find tensor library in {search_dirs}. Available files: {[f for d in search_dirs if os.path.exists(d) for f in os.listdir(d)]}")

lib = ctypes.CDLL(_get_lib_path())

class DType:
  FLOAT32, FLOAT64, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, BOOL = range(11)

class DTypeValue(ctypes.Union):
  _fields_ = [("f32", c_float), ("f64", c_double), ("i8", c_int8), ("i16", c_int16), ("i32", c_int32), ("i64", c_int64), ("u8", c_uint8), ("u16", c_uint16), ("u32", c_uint32), ("u64", c_uint64), ("boolean", c_uint8)]

class CTensor(Structure):
  _fields_ = [("data", c_void_p), ("strides", POINTER(c_int)), ("shape", POINTER(c_int)), ("size", c_size_t), ("ndim", c_size_t), ("dtype", c_int), ("is_view", c_int)]

def _setup_func(name, argtypes, restype):
  func = getattr(lib, name)
  func.argtypes, func.restype = argtypes, restype
  return func

_funcs = {
  'create_tensor': ([POINTER(c_float), c_size_t, POINTER(c_int), c_size_t, c_int], POINTER(CTensor)),
  'delete_tensor': ([POINTER(CTensor)], None), 'delete_data': ([POINTER(CTensor)], None),
  'delete_shape': ([POINTER(CTensor)], None), 'delete_strides': ([POINTER(CTensor)], None),
  'print_tensor': ([POINTER(CTensor)], None), 'out_data': ([POINTER(CTensor)], POINTER(c_float)),
  'out_shape': ([POINTER(CTensor)], POINTER(c_int)), 'out_strides': ([POINTER(CTensor)], POINTER(c_int)),
  'out_size': ([POINTER(CTensor)], c_int), 'contiguous_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'is_contiguous_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'make_contiguous_inplace_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'view_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'is_view_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'cast_tensor': ([POINTER(CTensor), c_int], POINTER(CTensor)), 'cast_tensor_simple': ([POINTER(CTensor), c_int], POINTER(CTensor)),
  'get_dtype_size': ([c_int], c_size_t), 'get_dtype_name': ([c_int], c_char_p),
  'dtype_to_float32': ([c_void_p, c_int, c_size_t], c_float), 'float32_to_dtype': ([c_float, c_void_p, c_int, c_size_t], None),
  'convert_to_float32': ([c_void_p, c_int, c_size_t], POINTER(c_float)), 'convert_from_float32': ([POINTER(c_float), c_void_p, c_int, c_size_t], None),
  'allocate_dtype_tensor': ([c_int, c_size_t], c_void_p), 'copy_with_dtype_conversion': ([c_void_p, c_int, c_void_p, c_int, c_size_t], None),
  'cast_tensor_dtype': ([c_void_p, c_int, c_int, c_size_t], c_void_p), 'is_integer_dtype': ([c_int], c_int),
  'is_float_dtype': ([c_int], c_int), 'is_unsigned_dtype': ([c_int], c_int), 'is_signed_dtype': ([c_int], c_int),
  'clamp_to_int_range': ([c_double, c_int], c_int64), 'clamp_to_uint_range': ([c_double, c_int], c_uint64),
  'get_dtype_priority': ([c_int], c_int), 'promote_dtypes': ([c_int, c_int], c_int),
  'add_tensor': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)), 'add_scalar_tensor': ([POINTER(CTensor), c_float], POINTER(CTensor)),
  'add_broadcasted_tensor': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)), 'sub_tensor': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)),
  'sub_scalar_tensor': ([POINTER(CTensor), c_float], POINTER(CTensor)), 'sub_broadcasted_tensor': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)),
  'mul_tensor': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)), 'mul_scalar_tensor': ([POINTER(CTensor), c_float], POINTER(CTensor)),
  'mul_broadcasted_tensor': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)), 'div_tensor': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)),
  'div_scalar_tensor': ([POINTER(CTensor), c_float], POINTER(CTensor)), 'div_broadcasted_tensor': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)),
  'pow_tensor': ([POINTER(CTensor), c_float], POINTER(CTensor)), 'pow_scalar': ([c_float, POINTER(CTensor)], POINTER(CTensor)),
  'log_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'exp_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'abs_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'matmul_tensor': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)),
  'batch_matmul_tensor': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)), 'broadcasted_matmul_tensor': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)),
  'sin_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'sinh_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'cos_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'cosh_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'tan_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'tanh_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'transpose_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'equal_tensor': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)),
  'reshape_tensor': ([POINTER(CTensor), POINTER(c_int), c_int], POINTER(CTensor)), 'squeeze_tensor': ([POINTER(CTensor), c_int], POINTER(CTensor)),
  'expand_dims_tensor': ([POINTER(CTensor), c_int], POINTER(CTensor)), 'flatten_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'sum_tensor': ([POINTER(CTensor), c_int, ctypes.c_bool], POINTER(CTensor)), 'min_tensor': ([POINTER(CTensor), c_int, ctypes.c_bool], POINTER(CTensor)),
  'max_tensor': ([POINTER(CTensor), c_int, ctypes.c_bool], POINTER(CTensor)), 'mean_tensor': ([POINTER(CTensor), c_int, ctypes.c_bool], POINTER(CTensor)),
  'var_tensor': ([POINTER(CTensor), c_int, c_int], POINTER(CTensor)), 'std_tensor': ([POINTER(CTensor), c_int, c_int], POINTER(CTensor)),
  'zeros_like_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'ones_like_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'zeros_tensor': ([POINTER(c_int), c_size_t, c_size_t, c_int], POINTER(CTensor)), 'ones_tensor': ([POINTER(c_int), c_size_t, c_size_t, c_int], POINTER(CTensor)),
  'randn_tensor': ([POINTER(c_int), c_size_t, c_size_t, c_int], POINTER(CTensor)), 'randint_tensor': ([c_int, c_int, POINTER(c_int), c_size_t, c_size_t, c_int], POINTER(CTensor)),
  'uniform_tensor': ([c_int, c_int, POINTER(c_int), c_size_t, c_size_t, c_int], POINTER(CTensor)), 'fill_tensor': ([c_float, POINTER(c_int), c_size_t, c_size_t, c_int], POINTER(CTensor)),
  'linspace_tensor': ([c_float, c_float, c_float, POINTER(c_int), c_size_t, c_size_t, c_int], POINTER(CTensor))
}

for name, (argtypes, restype) in _funcs.items(): _setup_func(name, argtypes, restype)