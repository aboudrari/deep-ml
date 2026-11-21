import numpy as np

def shuffle_data(X, y, seed=None):
	# Your code here
	if seed is not None:
		np.random.seed(seed)
		
	arr = np.arange(0 , len(X))    
	np.random.shuffle(arr)
	return X[arr] , y[arr]
        