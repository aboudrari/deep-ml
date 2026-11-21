import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	a = np.array(a)
	p_old , q_old = len(a) , len(a[0])
	p_new , q_new = new_shape[0] , new_shape[1]
	if p_old * q_old != p_new * q_new:
		return []
	else:
		return a.reshape(p_new,q_new).tolist()

