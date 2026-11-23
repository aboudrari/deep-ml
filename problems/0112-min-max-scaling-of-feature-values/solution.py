def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    # Your code here
    _min = min(x)
    _max = max(x)
    result = []
    for num in x :
        norml = (num - _min) / (_max - _min)
        result.append(norml)
    return result 