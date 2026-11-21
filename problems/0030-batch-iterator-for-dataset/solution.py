import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	# Your code here
	n = len(X)
	result = []
	for i in range(0, n, batch_size):
		X_batch = X[i:i+batch_size]
		
		if y is None:
			result.append(list(X_batch))
		else :
			y_batch = y[i:i+batch_size]
			result.append([X_batch,y_batch])
	return result 		
					

		