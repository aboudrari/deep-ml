import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Your code here
    indices = np.arange(n_samples) 
    if shuffle :
        np.random.shuffle(indices)
    folds = []
    start = 0
    base_size = n_samples // k
    remainder = n_samples %  k 
    for i in range(k):
        if i < remainder:
            fold_size = base_size + 1
        else:
            fold_size = base_size 
        end = start + fold_size
        folds.append(indices[start:end])
        start = end
    result = []
    for i in range(k):
        test_indices  = folds[i]        # just fold i
        train_indices = np.concatenate([folds[j] for j in range(k) if j != i])         # all other folds combined
        result.append((train_indices, test_indices))
    return result
        