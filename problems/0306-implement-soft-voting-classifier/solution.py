import numpy as np

def soft_voting_classifier(probabilities: np.ndarray, weights: list = None) -> list:
    if weights is None:
        weights = np.ones(len(probabilities))
	weights = np.array(weights)
    weights = weights / weights.sum()  

    avg_probs = np.average(probabilities, axis=0, weights=weights)

    result = []
    for row in avg_probs:
        result.append(np.argmax(row))
    return list(map(int, result))