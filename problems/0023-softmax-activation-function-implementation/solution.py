import math
import numpy as np 
def softmax(scores):
    maxx = np.max(scores)       # compute once, using ALL scores
    denom = np.sum([math.exp(z - maxx) for z in scores])       # compute once, using ALL scores
    
    results = []
    for score in scores:
        result = float(math.exp(score - maxx) / denom )   # now use maxx and denom here
        results.append(result)
    return results

