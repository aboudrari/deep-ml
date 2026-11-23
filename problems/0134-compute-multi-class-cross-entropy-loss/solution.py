import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    # 1. Collect the probability given to the true class of each sample
    ones_ = []
    for probs_row, label_row in zip(predicted_probs, true_labels):
        position = np.argmax(label_row)           # where the 1 sits in the one-hot row
        ones_.append(probs_row[position])      # prob of the correct class

    # 2. Clip for numerical stability (avoid log(0) = -inf)
    p = np.clip(np.array(ones_), epsilon, 1 - epsilon)

    # 3. Loss = -mean(log(p_correct))
    return -np.mean(np.log(p))