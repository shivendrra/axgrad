import math

type_dtypes = ["int8", "int16", "int32", "int64", "long", "float32", "float64", "double", "uint8", "uint16", "uint32", "uint64", "bool"]

get_dtypes:list = lambda: type_dtypes
get_shape:list = lambda data: [len(data)] + get_shape(data[0]) if isinstance(data, list) else []
flatten:list = lambda subdata: [item for sub in subdata for item in flatten(sub)] if isinstance(subdata, list) else [subdata]
get_size:int = lambda shape: 1 if not shape else shape[0] * get_size(shape[1:])
get_strides:list = lambda shape: [1] if len(shape) <= 1 else get_strides(shape[1:]) + [get_strides(shape[1:])[0] * shape[-1]]
transposed_shape:list = lambda shape: shape if len(shape) == 1 else [shape[1], shape[0]] if len(shape) == 2 else [shape[0], shape[2], shape[1]] if len(shape) == 3 else (_ for _ in ()).throw(ValueError(f"Unsupported shape dimension: {len(shape)}"))
reshape_list:list = lambda flat_list, shape: flat_list[:shape[0]] if len(shape) == 1 else [reshape_list(flat_list[i * (len(flat_list) // shape[0]):(i + 1) * (len(flat_list) // shape[0])], shape[1:]) for i in range(shape[0])]