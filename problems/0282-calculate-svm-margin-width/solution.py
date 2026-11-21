import numpy as np

def svm_margin_width(w: np.ndarray) -> float:
    """
    Calculate the margin width of a linear SVM classifier.
    
    Parameters:
    w : np.ndarray - weight vector defining the hyperplane
    
    Returns:
    float - the total margin width
    """
    n = len(w)
    sqrts = []
    for i in w :
        sqrts.append(i**2)
    return 2 / np.sqrt(sum(sqrts)) 