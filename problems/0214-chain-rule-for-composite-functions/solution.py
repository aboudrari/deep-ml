import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	"""
	Compute derivative of composite functions using chain rule.
	
	Args:
		functions: List of function names (applied right to left)
		          Available: 'square', 'sin', 'exp', 'log'
		x: Point at which to evaluate derivative
	
	Returns:
		Derivative value at x
	
	Example:
		['sin', 'square'] represents sin(x²)
		['exp', 'sin', 'square'] represents exp(sin(x²))
	"""
	# Your code here
	# Forward pass 
	derivs = {
    'square': lambda x: 2*x,
    'sin': np.cos,
    'exp': np.exp,
    'log': lambda x: 1/x,
}
	funcs = {
    'square': lambda x: x**2,
    'sin': np.sin,
    'exp': np.exp,
    'log': np.log,
}
	u = x
	intermediates = [x]
	for f in reversed(functions):
		u = funcs[f](u)
		intermediates.append(u)
    
    # back pass 
	result = 1.0
	for i , f in enumerate(reversed(functions)):
		result *= derivs[f](intermediates[i])
	return round(result , 6)
		
    
		
       

	