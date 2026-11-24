import numpy as np

def softmax(values):
    # Implement the softmax function
    exp_values = np.exp(values)
    denominator = np.sum(exp_values)
    return exp_values / denominator

def pattern_weaver(n, crystal_values, dimension):
    # Your code here
    outputs = []

    for i, v_i in enumerate(crystal_values):
        scores = []
        for j, v_j in enumerate(crystal_values):
            score = (v_i * v_j) / np.sqrt(dimension)
            scores.append(score)

        attention_weights = softmax(scores)

        weighted_sum = np.dot(crystal_values, attention_weights)
        
        outputs.append(weighted_sum)

    return np.round(outputs, 4)

