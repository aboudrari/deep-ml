import numpy as np

def precision_recall_at_threshold(y_true, y_scores, threshold):
    """
    Compute precision and recall at a given decision threshold.

    Args:
        y_true: list/array of true binary labels (0 or 1)
        y_scores: list/array of predicted scores in [0, 1]
        threshold: float, classification threshold (predict positive if score >= threshold)

    Returns:
        [precision, recall] as a list of two floats rounded to 4 decimals.
    """
    # Your code here
    TP = 0 
    FP = 0 
    FN = 0
    for i in range(len(y_true)):
        if y_scores[i] >= threshold :
            if y_true[i] == 1 :
                TP += 1
            else : #y_true[i] == 0 
                FP += 1
        else :
            if y_true[i] == 1 :
                FN += 1
    precision = 0.0 if (TP + FP) == 0 else TP / (TP + FP)
    recall = 0.0 if (TP + FN) == 0 else TP / (TP + FN)

    return [round(precision, 4), round(recall, 4)]
                
