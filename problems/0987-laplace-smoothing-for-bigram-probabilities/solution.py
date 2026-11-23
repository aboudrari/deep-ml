import numpy as np

def smooth_bigram_probs(N, k):
    """
    N: 2D list or array of bigram counts (V x V)
    k: float, add-k smoothing constant (k >= 0)
    Returns: 2D list of smoothed bigram probabilities (V x V).
    """
    N = np.array(N)
    smoothed = N + k
    row_sums = np.sum(smoothed, axis=1, keepdims=True)
    P = smoothed / row_sums
    return  P.tolist()



