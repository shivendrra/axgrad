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
          if file.startswith(name) and any(file.endswith(ext) for ext in possible_exts if ext): return os.path.join(root, file)
  raise FileNotFoundError(f"Could not find tensor library in {search_dirs}. Available files: {[f for d in search_dirs if os.path.exists(d) for f in os.listdir(d)]}")

lib = ctypes.CDLL(_get_lib_path())

class DType: FLOAT32, FLOAT64, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, BOOL = range(11)
class DTypeValue(ctypes.Union): _fields_ = [("f32", c_float), ("f64", c_double), ("i8", c_int8), ("i16", c_int16), ("i32", c_int32), ("i64", c_int64), ("u8", c_uint8), ("u16", c_uint16), ("u32", c_uint32), ("u64", c_uint64), ("boolean", c_uint8)]
class CTensor(Structure):  _fields_ = [("data", c_void_p), ("strides", POINTER(c_int)), ("shape", POINTER(c_int)), ("size", c_size_t), ("ndim", c_size_t), ("dtype", c_int), ("is_view", c_int)]
def _setup_func(name, argtypes, restype):
  func = getattr(lib, name)
  func.argtypes, func.restype = argtypes, restype
  return func

_forward_funcs = {
  'create_tensor': ([POINTER(c_float), c_size_t, POINTER(c_int), c_size_t, c_int], POINTER(CTensor)),
  'delete_tensor': ([POINTER(CTensor)], None), 'delete_data': ([POINTER(CTensor)], None),
  'delete_shape': ([POINTER(CTensor)], None), 'delete_strides': ([POINTER(CTensor)], None),
  'print_tensor': ([POINTER(CTensor)], None), 'out_data': ([POINTER(CTensor)], POINTER(c_float)),
  'out_shape': ([POINTER(CTensor)], POINTER(c_int)), 'out_strides': ([POINTER(CTensor)], POINTER(c_int)),
  'out_size': ([POINTER(CTensor)], c_int), 'contiguous_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'get_item_tensor': ([POINTER(CTensor), POINTER(c_int)], c_float), 'set_item_tensor': ([POINTER(CTensor), POINTER(c_int), c_float], None),
  'get_linear_index': ([POINTER(CTensor), POINTER(c_int)], c_int),
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
  'log_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'exp_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'sign_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'neg_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'sqrt_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'abs_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'matmul_tensor': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)),
  'batch_matmul_tensor': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)), 'broadcasted_matmul_tensor': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)),
  'sin_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'sinh_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'cos_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'cosh_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'tan_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'tanh_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'sigmoid_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'relu_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'gelu_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'silu_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'leaky_relu_tensor': ([POINTER(CTensor), c_float], POINTER(CTensor)), 'elu_tensor': ([POINTER(CTensor), c_float], POINTER(CTensor)),
  'softplus_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'swish_tensor': ([POINTER(CTensor), c_float], POINTER(CTensor)),
  'transpose_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'equal_tensor': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)), 'not_equal_tensor': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)),
  'equal_scalar': ([POINTER(CTensor), c_float], POINTER(CTensor)), 'not_equal_scalar': ([POINTER(CTensor), c_float], POINTER(CTensor)),
  'greater_tensor': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)), 'smaller_tensor': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)),
  'greater_scalar': ([POINTER(CTensor), c_float], POINTER(CTensor)), 'smaller_scalar': ([POINTER(CTensor), c_float], POINTER(CTensor)),
  'greater_equal_tensor': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)), 'smaller_equal_tensor': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)),
  'greater_equal_scalar': ([POINTER(CTensor), c_float], POINTER(CTensor)), 'smaller_equal_scalar': ([POINTER(CTensor), c_float], POINTER(CTensor)),
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

_backward_funcs = {
  'sin_backwards': ([POINTER(CTensor)], POINTER(CTensor)), 'cos_backwards': ([POINTER(CTensor)], POINTER(CTensor)), 'tan_backwards': ([POINTER(CTensor)], POINTER(CTensor)),
  'sinh_backwards': ([POINTER(CTensor)], POINTER(CTensor)), 'cosh_backwards': ([POINTER(CTensor)], POINTER(CTensor)), 'tanh_backwards': ([POINTER(CTensor)], POINTER(CTensor)),
  'relu_backwards': ([POINTER(CTensor)], POINTER(CTensor)), 'sigmoid_backwards': ([POINTER(CTensor)], POINTER(CTensor)),
  'gelu_backwards': ([POINTER(CTensor)], POINTER(CTensor)), 'silu_backwards': ([POINTER(CTensor)], POINTER(CTensor)), 'softplus_backwards': ([POINTER(CTensor)], POINTER(CTensor)),
  'leaky_relu_backwards': ([POINTER(CTensor), c_float], POINTER(CTensor)), 'elu_backwards': ([POINTER(CTensor), c_float], POINTER(CTensor)), 'swish_backwards': ([POINTER(CTensor), c_float], POINTER(CTensor)),
  'sum_backwards': ([POINTER(CTensor), POINTER(c_int), c_int, c_size_t, c_int], POINTER(CTensor)), 'mean_backwards': ([POINTER(CTensor), POINTER(c_int), c_int, c_size_t, c_int], POINTER(CTensor)),
  'var_backwards': ([POINTER(CTensor), POINTER(CTensor), POINTER(c_int), c_int, c_size_t, c_int, c_int], POINTER(CTensor)), 'std_backwards': ([POINTER(CTensor), POINTER(CTensor), POINTER(c_int), c_int, c_size_t, c_int, c_int], POINTER(CTensor)),
  'clip_backwards': ([POINTER(CTensor), POINTER(CTensor), c_float], POINTER(CTensor)), 'clamp_backwards': ([POINTER(CTensor), POINTER(CTensor), c_float, c_float], POINTER(CTensor)),
  'mm_norm_backwards': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)), 'rms_norm_backwards': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)), 'std_norm_backwards': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)),
  'l1_norm_backwards': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)), 'l2_norm_backwards': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)),
  'unit_norm_backwards': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)), 'robust_norm_backwards': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)),
  'log_backwards': ([POINTER(CTensor)], POINTER(CTensor)), 'exp_backwards': ([POINTER(CTensor)], POINTER(CTensor)),
  'abs_backwards': ([POINTER(CTensor)], POINTER(CTensor)), 'sqrt_backwards': ([POINTER(CTensor)], POINTER(CTensor)),
}

_nn_funcs = {
  'clip_tensor': ([POINTER(CTensor), c_float], POINTER(CTensor)), 'clamp_tensor': ([POINTER(CTensor), c_float, c_float], POINTER(CTensor)),
  'mm_norm_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'std_norm_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'rms_norm_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'unit_norm_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'l1_norm_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'l2_norm_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'robust_norm_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
}

_vector_funcs = {
  'vector_dot': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)), 'vector_matrix_dot': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)),
  'vector_inner': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)), 'vector_outer': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)),
  'vector_cross': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)), 'vector_cross_axis': ([POINTER(CTensor), POINTER(CTensor), c_int], POINTER(CTensor)),
  'det_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'batched_det_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'inv_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'solve_tensor': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)), 'lstsq_tensor': ([POINTER(CTensor), POINTER(CTensor)], POINTER(CTensor)),
  'eig_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'batched_eig_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'eigv_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'batched_eigv_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'eigh_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'batched_eigh_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'eighv_tensor': ([POINTER(CTensor)], POINTER(CTensor)), 'batched_eighv_tensor': ([POINTER(CTensor)], POINTER(CTensor)),
  'svd_tensor': ([POINTER(CTensor)], POINTER(POINTER(CTensor))), 'qr_tensor': ([POINTER(CTensor)], POINTER(POINTER(CTensor))), 'batched_qr_tensor': ([POINTER(CTensor)], POINTER(POINTER(CTensor))),
  'lu_tensor': ([POINTER(CTensor)], POINTER(POINTER(CTensor))), 'batched_lu_tensor': ([POINTER(CTensor)], POINTER(POINTER(CTensor))),
}

for name, (argtypes, restype) in _forward_funcs.items(): _setup_func(name, argtypes, restype)
for name, (argtypes, restype) in _backward_funcs.items(): _setup_func(name, argtypes, restype)
for name, (argtypes, restype) in _nn_funcs.items(): _setup_func(name, argtypes, restype)
for name, (argtypes, restype) in _vector_funcs.items(): _setup_func(name, argtypes, restype)