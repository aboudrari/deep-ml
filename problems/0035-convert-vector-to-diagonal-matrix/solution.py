import numpy as np

def make_diagonal(x):
	# Your code here
	x = np.array(x, dtype=float)
	result = np.diag(x)
	return result

