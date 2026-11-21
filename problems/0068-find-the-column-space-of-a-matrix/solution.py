import numpy as np

def matrix_image(A):
    A = A.astype(float)          # work with floats
    m, n = A.shape
    R = A.copy()

    pivot_cols = []
    row = 0

    for col in range(n):
        # Find a pivot row
        pivot = None
        for r in range(row, m):
            if abs(R[r, col]) > 1e-10:
                pivot = r
                break

        if pivot is None:
            continue

        # Swap rows
        R[[row, pivot]] = R[[pivot, row]]

        # Eliminate below
        for r in range(row + 1, m):
            R[r] -= (R[r, col] / R[row, col]) * R[row]

        pivot_cols.append(col)
        row += 1

        if row == m:
            break

    # Extract independent columns from ORIGINAL matrix
    return A[:, pivot_cols].astype(int)