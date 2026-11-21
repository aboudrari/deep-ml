import numpy as np 

def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
        
        def determinant(matrix):
            if len(matrix) == 1:
                return matrix[0][0] 
            det = 0
            for j in range(len(matrix)):
                minor = [row[:j] + row[j+1:] for row in matrix[1:]]
                det += (-1)**(j) * matrix[0][j] * determinant(minor)
            return det	
        
        the_trace = sum(np.diag(matrix))
        the_det = determinant(matrix)
        return (float(the_det), float(the_trace))
        