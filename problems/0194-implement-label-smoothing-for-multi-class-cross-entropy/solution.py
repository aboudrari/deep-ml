import numpy as np

def smooth_labels(y_true, num_classes, epsilon):

    if epsilon < 0 or epsilon > 1 :
        raise ValueError("epsilon must be between 0 and 1")

    N = len(y_true)

    other_classes = epsilon / num_classes
    true_class = 1 - epsilon + other_classes 

    classes = np.full((N, num_classes), other_classes)
    row_indices = np.arange(len(y_true))
    classes[row_indices, y_true] = true_class
    
    return classes



def label_smoothing_cross_entropy(logits, y_true, num_classes, epsilon=0.1, round_decimals=None):

    logits = np.asarray(logits)
    y_true = np.asarray(y_true)
    
    if logits.ndim != 2 or logits.shape[1] != num_classes:
        raise ValueError("Invalid logits shape")

    max_logits = np.max(logits, axis=1, keepdims=True)
    shifted_logits = logits - max_logits
    exp_logits = np.exp(shifted_logits)
    sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
    log_sum_exp = np.log(sum_exp)
    log_softmax = shifted_logits - log_sum_exp

    smoothed_labels = smooth_labels(y_true, num_classes, epsilon)

    loss_per_sample = -np.sum(
        smoothed_labels * log_softmax,
        axis=1)
    
    loss = np.mean(loss_per_sample)

    return loss

