import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
    A , T , S = np.array(A) , np.array(T) , np.array(S)
    def is_invertable(X):
        m , n = X.shape
        if m != n or np.linalg.det(X) == 0.0:
            return False 
        return True
    if not is_invertable(S) or not is_invertable(T):
        return -1 
    transform_matrix = np.linalg.inv(T) @ A @ S

    return transform_matrix
	