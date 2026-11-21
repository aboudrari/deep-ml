import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	# Your code here
	gradient = np.array(gradient)
	Magnitude = np.linalg.norm(gradient)
	if Magnitude == 0:
    	return {
        'magnitude': 0.0,
        'direction': [0.0] * len(gradient),
        'descent_direction': [0.0] * len(gradient)
    }
	Direction = gradient / Magnitude
	Descent_direction = -Direction
	return {'magnitude': Magnitude, 'direction': Direction.tolist(), 'descent_direction': Descent_direction.tolist()}

