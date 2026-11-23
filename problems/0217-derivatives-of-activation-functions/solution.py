import numpy as np 
import math 

def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	# Your code here
	# {'sigmoid': 0.25, 'tanh': 1.0, 'relu': 0.0} 
	def sigmoid_dev(x):
		# 
		sigmoid = (1 / (1 + np.exp(-x)))
		return float(sigmoid * (1 - sigmoid))
	
	def tanh_dev(x):
		tanh = ((math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)))
		return float(1 - tanh**2)
	
	def relu_dev(x):
		if x > 0 :
			return 1.0
		else:
			return 0.0
	
	return {'sigmoid': sigmoid_dev(x), 'tanh': tanh_dev(x), 'relu': relu_dev(x)}

