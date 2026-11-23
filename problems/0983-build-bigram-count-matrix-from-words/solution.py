import numpy as np 

def bigram_counts(words):
    all_chars = sorted(set(''.join(words)))
    vocab = ['.'] + all_chars
    stoi = {ch: i for i, ch in enumerate(vocab)}
    N = np.zeros((len(vocab), len(vocab)), dtype=np.int32)
    
    for w in words:
        sequence = ['.'] + list(w) + ['.']
        for t in range(len(sequence) - 1):
            N[stoi[sequence[t]], stoi[sequence[t+1]]] += 1
    
    return N.tolist()
