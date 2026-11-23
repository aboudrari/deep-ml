import numpy as np

def stratified_train_test_split(X, y, test_size, random_seed=None):
	
	np.random.seed(random_seed)
	classes = np.unique(y)
	all_train_indices = []
	all_test_indices = []
	for cls  in classes:
		indices = np.where(y == cls)[0]  
		# shuffle the indices 
		np.random.shuffle(indices)

		n_class_samples = len(indices)
		n_test = int(n_class_samples * test_size)

		test_indices = indices[:n_test]
		train_indices = indices[n_test:]

		all_train_indices.extend(train_indices) 
		all_test_indices.extend(test_indices)

	X_train = X[all_train_indices]
	X_test = X[all_test_indices]
	y_train = y[all_train_indices]
	y_test = y[all_test_indices]
	
	return X_train, X_test, y_train, y_test


