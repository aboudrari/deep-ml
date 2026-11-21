import math
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	
	a , b , c , d  = matrix[0][0] , matrix[0][1] , matrix[1][0] , matrix[1][1]
	det = a * d - b * c
	trace = a + d
	lambda1 = (trace + math.sqrt(trace**2 - 4*det)) / 2
	lambda2 = (trace - math.sqrt(trace**2 - 4*det)) / 2
 
	return sorted([lambda1, lambda2], reverse=True)

