def matrix_mul(A, B):
    """
    Performs matrix multiplication of matrices A and B.
    
    Args:
    - A: First matrix (2D list)
    - B: Second matrix (2D list)
    
    Returns:
    - Result of matrix multiplication (2D list)
    """
    # Check if matrices are compatible for multiplication
    if len(A[0]) != len(B):
        raise ValueError("Matrices are not compatible for multiplication")

    # Initialize result matrix with zeros
    result = [[0] * len(B[0]) for _ in range(len(A))]

    # Perform matrix multiplication
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]

    return result

# Example usage
A = [[1, 2, 3],
     [4, 5, 6]]

B = [[7, 8],
     [9, 10],
     [11, 12]]

result = matrix_mul(A, B)
print("Result of matrix multiplication:", result)