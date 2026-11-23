import numpy as np

def bigram_avg_nll(P, words, stoi):
    """Average negative log-likelihood of words under a bigram model."""
    N = 0
    log_likelihood  = 0.0
    for word in words :
        
        chs = ["."] + list((word)) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
           p = P[stoi[ch1], stoi[ch2]]  
           log_likelihood += np.sum(np.log(p))
           N += 1
    if N == 0:
        return 0.0
    return round(-log_likelihood / N, 4)

