def get_shape(data):
  return (len(data), ) + get_shape(data[0]) if isinstance(data, list) else ()

def _flatten(data):
  return [item for sublist in data for item in _flatten(sublist)] if isinstance(data, list) else [data]

