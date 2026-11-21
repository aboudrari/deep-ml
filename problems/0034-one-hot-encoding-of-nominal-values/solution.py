import numpy as np

def to_categorical(x, n_col=None):
	# Your code here
	if not n_col :
		n_col = len(np.unique(x))
		
	output = np.zeros((len(x), n_col))
	for row , column  in enumerate(x):
		output[row][column] = 1
		
	return output 
