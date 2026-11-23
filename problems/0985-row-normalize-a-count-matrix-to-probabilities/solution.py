import numpy as np

def row_normalize(counts: list[list[float]]) -> list[list[float]]:
	"""Convert a count matrix into a row-stochastic probability matrix."""
	counts = np.array(counts)
	row_sums = counts.sum(axis=1, keepdims=True)  # sum of each row 
	safe_row_sums = row_sums.copy()
	safe_row_sums[row_sums == 0] = 1
	result = counts / safe_row_sums 
	return result.tolist()
