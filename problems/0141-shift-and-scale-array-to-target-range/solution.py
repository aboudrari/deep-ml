import numpy as np

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    """
    Shift and scale values from their original range [min, max] to a target [c, d] range.
    """
    # Your code here
    a = np.min(values)
    b = np.max(values)
    if a == b :
        return np.full(values.shape, c)
    else :
        f = c + ((d - c) / (b - a))*(values - a)
        return  f.astype(float)
