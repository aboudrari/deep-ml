def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    n = len(vectors)        # number of features
    m = len(vectors[0])     # number of observations
    
    # step 1: calculate mean for each feature
    means = [sum(v) / len(v) for v in vectors]
    
    # step 2: initialize n x n matrix of zeros
    cov_matrix = [[0]*n for _ in range(n)]
    
    # step 3: nested loop over each pair (i, j)
    for i in range(n):
        for j in range(n):
            cov_matrix[i][j] = sum((vectors[i][k] - means[i]) * (vectors[j][k] - means[j]) for k in range(m)) / (m - 1)
    
    return cov_matrix