class Array:
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], list):
            self.data = args[0]
            self.shape = (len(args[0]),)
        else:
            self.data = list(args)
            self.shape = (len(args),)

    def __repr__(self):
        return f"Array({self.data})"

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __add__(self, other):
        if isinstance(other, Array):
            return Array([x + y for x, y in zip(self.data, other.data)])
        else:
            return Array([x + other for x in self.data])

    def __sub__(self, other):
        if isinstance(other, Array):
            return Array([x - y for x, y in zip(self.data, other.data)])
        else:
            return Array([x - other for x in self.data])

    def __mul__(self, other):
        if isinstance(other, Array):
            return Array([x * y for x, y in zip(self.data, other.data)])
        else:
            return Array([x * other for x in self.data])

    def __truediv__(self, other):
        if isinstance(other, Array):
            return Array([x / y for x, y in zip(self.data, other.data)])
        else:
            return Array([x / other for x in self.data])

# Example usage
arr1 = Array(1, 2, 3)
arr2 = Array([4, 5, 6])

print("Array 1:", arr1)
print("Array 2:", arr2)

print("Addition:", arr1 + arr2)
print("Subtraction:", arr2 - arr1)
print("Multiplication:", arr1 * 2)
print("Division:", arr2 / arr1)