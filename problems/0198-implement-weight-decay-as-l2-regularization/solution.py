def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
	"""
	Apply weight decay (L2 regularization) to parameters.
	
	Args:
		parameters: List of parameter arrays
		gradients: List of gradient arrays
		lr: Learning rate
		weight_decay: Weight decay factor
		apply_to_all: Boolean list indicating which parameter groups get weight decay
	
	Returns:
		Updated parameters
	"""
	# Your code here
	
	for i in range(len(parameters)):
		if apply_to_all[i]: 
			for j in range(len(gradients[i])):
				# wnew​=w−η⋅g−η⋅λ⋅w 
				w_new = parameters[i][j] - (lr * gradients[i][j]) - (lr * weight_decay * parameters[i][j] )
				parameters[i][j] = w_new
		else :
			for j in range(len(gradients[i])): 
				# w - lr * g 
				w_new = parameters[i][j] - (lr * gradients[i][j])
				parameters[i][j] = w_new
	return parameters