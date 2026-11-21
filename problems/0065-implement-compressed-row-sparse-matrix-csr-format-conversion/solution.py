import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

	:param dense_matrix: 2D list representing a dense matrix
	:return: A tuple containing (values array, column indices array, row pointer array)
	"""
	value_arr = []
	col_index = []
	row_ptr = [0]
	for row in dense_matrix :
		for index, value in enumerate(row):
			if value != 0 :
				value_arr.append(value)
				col_index.append(index)
		row_ptr.append(len(value_arr))

	return value_arr , col_index , row_ptr
	