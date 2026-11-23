import torch
import numpy as np


def fit_linear_regression(X, y, lr=0.1, steps=500):
	"""Fit y ~= X @ w + b with full-batch GD using only autograd.

	Args:
		X: Float tensor (N, D)
		y: Float tensor (N,) or (N, 1)
		lr: learning rate
		steps: number of gradient descent iterations

	Returns:
		w: Float tensor (D,) learned weights (no grad)
		b: Float tensor scalar learned bias (no grad)
	"""

	w = torch.zeros(X.shape[1], requires_grad=True)
	b = torch.tensor(0.0, requires_grad=True)

	for _ in range(steps):
		y_pred = X @ w + b 
		loss = torch.mean((y_pred - y)**2)
		loss.backward()
		#   - manual GD update under torch.no_grad()
		with torch.no_grad():
			w -= lr * w.grad
			b -= lr * b.grad
		#   - zero gradients
		w.grad.zero_()
		b.grad.zero_()
		
	# return detached w, b
	return w.detach(), b.detach()
	
