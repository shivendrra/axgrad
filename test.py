a = [[1, 4, 6], [1, 5, 8]]
b = [[[1, 4, 6], [1, 5, 8]],
     [[1, 4, 6], [1, 5, 8]],
     [[1, 4, 6], [1, 5, 8]],
     [[1, 4, 6], [1, 5, 8]]]

def get_shape(data):
  return (len(data), ) + get_shape(data[0]) if isinstance(data, list) else ()

def flatten(input_tensor, start_dim=0, end_dim=-1):
    # Helper function to recursively flatten nested lists
    def _flatten(data):
        if isinstance(data, list):
            result = []
            for item in data:
                result.extend(_flatten(item))
            return result
        else:
            return [data]

    # Handle the flattening according to the specified dimensions
    def _recurse_flatten(data, current_dim):
        if current_dim < start_dim:
            return [_recurse_flatten(item, current_dim + 1) for item in data]
        elif start_dim <= current_dim <= end_dim:
            return _flatten(data)
        else:
            return data

    # Calculate the end dimension based on the given end_dim
    if end_dim == -1:
        end_dim = len(input_tensor) - 1

    return _recurse_flatten(input_tensor, 0)

from axgrad import tensor

tensor1 = tensor([[1, 2, 3], [4, 5, 6]])
tensor2 = tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

print(tensor1.flatten())
print(tensor2.flatten())
print(tensor1.flatten(0, -1))
print(tensor2.flatten(1, 1))