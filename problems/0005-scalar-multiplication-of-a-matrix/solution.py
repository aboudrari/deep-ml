def scalar_multiply(matrix, scalar):
    result = []
    
    for row in matrix:
        new_row = []              # this will store the scaled elements of the current row
        
        for element in row:
            new_row.append(element * scalar)
        
        result.append(new_row)
    
    return result
