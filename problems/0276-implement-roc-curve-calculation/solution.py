import numpy as np

def compute_roc_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute ROC curve points (FPR, TPR) for binary classification.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Predicted scores/probabilities for the positive class
    
    Returns:
        Tuple of (fpr, tpr) where each is a list of floats
    """
    # Your code here
    result = []
    thresholds = [float('inf')] + sorted(set(y_scores), reverse=True)
    N = sum([1  for i in y_true if i == 0])
    P = sum([1  for i in y_true if i == 1])
    for threshold in thresholds :
        TP = 0
        FP = 0
        for i in range(len(y_scores)) :
            if y_scores[i] >= threshold :
                if y_true[i] == 1 :
                    TP += 1
                else :
                    FP += 1
        TPR = TP / P 
        FPR = FP / N 
        result.append((TPR,FPR))
    tpr_list, fpr_list = zip(*result)
    return list(fpr_list) , list(tpr_list)

