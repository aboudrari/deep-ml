import numpy as np

def lr_range_finder(X, y, w0, a, b, n_steps):
	"""
	Sweep learning rate from 10**a to 10**b over n_steps full-batch GD updates
	on a linear regression model (no bias). Return the list of MSE losses after
	each update.
	"""
	losses = []
	w = w0
	for k in range(n_steps):
		if n_steps > 1 :
			pw = a + (((b-a)*k)/(n_steps - 1))
			lr = 10**pw
		else :
			lr = 10**a
		# Prediction
		pred = X @ w
		# Error
		error = pred - y
		n = len(X)
		# Gradient
		gradient = (2/n) * X.T @ error 
		# Update
		w = w - lr * gradient
		# new prediction 
		new_pred = X @ w
		# loss 
		loss = np.mean((new_pred - y) ** 2)
		losses.append(float(loss))


	return losses

						
