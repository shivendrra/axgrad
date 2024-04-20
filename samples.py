x = [[[[2, 4, 4], [-2, 4, -4], [2, 4, 4], [-2, 4, -4]],
     [[2, 2, 3], [1, 2, 5], [2, 4, 4], [-2, 4, -4]],
     [[1, 6, -4], [2, -4, 4], [2, 4, 4], [-2, 4, -4]],
     [[1, 6, -4], [2, -4, 4], [2, 4, 4], [-2, 4, -4]],],
     [[[2, 4, 4], [-2, 4, -4], [2, 4, 4], [-2, 4, -4]],
     [[2, 2, 3], [1, 2, 5], [2, 4, 4], [-2, 4, -4]],
     [[1, 6, -4], [2, -4, 4], [2, 4, 4], [-2, 4, -4]],
     [[1, 6, -4], [2, -4, 4], [2, 4, 4], [-2, 4, -4]],],]

y = [[[[2, 4, 1], [-2, 4, 3], [-3, 4, 3]],
     [[2, 4, 1], [-2, 4, 3], [-3, 4, 3]],
     [[1, 6, 5], [2, -4, 5], [-1, -3, 9]],
     [[2, 4, 1], [-2, 4, 3], [-3, 4, 3]],],
     [[[2, 4, 1], [-2, 4, 3], [-3, 4, 3]],
     [[2, 4, 1], [-2, 4, 3], [-3, 4, 3]],
     [[1, 6, 5], [2, -4, 5], [-1, -3, 9]],
     [[2, 4, 1], [-2, 4, 3], [-3, 4, 3]],]]

z = [[[2, 4, 1], [-2, 4, 3], [-3, 4, 3]],
     [[2, 4, 1], [-2, 4, 3], [-3, 4, 3]],
     [[1, 6, 5], [2, -4, 5], [-1, -3, 9]],
     [[2, 4, 1], [-2, 4, 3], [-3, 4, 3]],]

a = [[2, 4, 4], [1, 5, 6]]
b = [[2, 4], [1, 5], [-1, 5]]

def convolution(arr1, arr2):
    m = len(arr1)
    n = len(arr2)
    result = [0] * (m + n - 1)

    for i in range(m):
        for j in range(n):
            result[i + j] += arr1[i] * arr2[j]

    return result

arr1 = [1, 2, 3]
arr2 = [4, 5]

conv_result = convolution(arr1, arr2)
print("Convolution result:", conv_result)

def convolution_2d(image, kernel):
    image_height, image_width = len(image), len(image[0])
    kernel_height, kernel_width = len(kernel), len(kernel[0])

    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    output = [[0] * output_width for _ in range(output_height)]

    for i in range(output_height):
        for j in range(output_width):
            for k in range(kernel_height):
                for l in range(kernel_width):
                    output[i][j] += image[i+k][j+l] * kernel[k][l]

    return output

image = list([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

kernel = list([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

result = convolution_2d(image, kernel)
print("Convolution result:", result)