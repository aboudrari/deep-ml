import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
	x = np.zeros(len(A))
	for epoch in range(n):
		new_x = np.zeros(len(x))
		for i in range(len(A)): 
			sum_ = 0
			for j in range(len(A)):
				if i == j :
					continue
				else :
					sum_ += A[i][j] * x[j]
			new_x[i] = (1 / A[i][i] )*( b[i] - sum_)
			new_x[i] = round(new_x[i], 4)
		x = new_x


	return x.tolist()