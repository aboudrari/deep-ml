def compressed_col_sparse_matrix(dense_matrix):
    """
    Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

    :param dense_matrix: List of lists representing the dense matrix
    :return: Tuple of (values, row indices, column pointer)
    """
    values = []
    row_index = []
    col_ptr = [0]
    for j in range(len(dense_matrix[0])):        # outer loop: walk through columns
        for i in range(len(dense_matrix)):       # inner loop: walk through rows
            value = dense_matrix[i][j]
            if value != 0:
                values.append(value)
                row_index.append(i)
        col_ptr.append(len(values))

    return values, row_index, col_ptr