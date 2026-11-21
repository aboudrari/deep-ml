import numpy as np

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    if len(v1) == 0 or len(v2) == 0:
        raise ValueError("Vectors cannot be empty")
    if v1.shape != v2.shape:
        raise ValueError("Vectors must have the same shape")
    
    dot_product = np.dot(v1, v2)
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    return dot_product / magnitude