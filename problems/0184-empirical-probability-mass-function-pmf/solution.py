import numpy as np 

def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    if not samples :
        return []
    n = len(samples)
    values, counts = np.unique(samples, return_counts=True) # return how many time the value is appears 
    prob = counts / n
    return [(int(v), float(p)) for v, p in zip(values, prob)]



    