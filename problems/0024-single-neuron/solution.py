import math
import numpy as np 


def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) :
	# Your code here
	z=np.dot(features,weights)+bias
	sigma = 1 / (1 + np.exp(-z))
	mse = np.mean((sigma - labels)**2)
	
	return np.round(sigma,4).tolist(), np.round(mse,4).tolist()

