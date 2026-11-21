import numpy as np 
def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	# P=C −1⋅B
	B = np.array(B)
	C = np.array(C)
	P = np.linalg.inv(C) @ B
	return np.round(P.tolist() , 4)

