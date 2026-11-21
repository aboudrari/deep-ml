def matrixmul(a:list[list[int|float]],b:list[list[int|float]])-> list[list[int|float]]:
	# Two matrices align for multiplication when: 
        # The number of columns in matrix A equals the number of rows in matrix B
    if len(a[0]) != len(b):
        return -1
    rows = len(a)
    cols = len(b[0])
    c = [[0] * cols for _ in range(rows)]
     
    for i in range(len(a)):        # rows of A
        for j in range(len(b[0])): # columns of B
            for k in range(len(b)): # accumulate
                c[i][j] += a[i][k] * b[k][j]

    return c


