a = [[1, 4, 6], [1, 5, 8]]
b = [[[1, 4, 6], [1, 5, 8]],
     [[1, 4, 6], [1, 5, 8]],
     [[1, 4, 6], [1, 5, 8]],
     [[1, 4, 6], [1, 5, 8]]]

def flatten(data):
  return [item for sublist in data for item in flatten(sublist)] if isinstance(data, list) else [data]

print(flatten(b))