import  math 
import  numpy as np 

def swish(x: float) -> float:
	"""
	Implements the Swish activation function.

	Args:
		x: Input value

	Returns:
		The Swish activation value
	"""
	# Your code here
	def sigmoid(z):
		return  1 / (1 + math.exp(-z))
	swish = x * sigmoid(x)
	return  swish