import numpy as np

def calculate_auc(y_true, y_scores):
    """
    Calculate the Area Under the ROC Curve (AUC).
    
    Args:
        y_true: List or array of binary ground truth labels (0 or 1)
        y_scores: List or array of predicted probabilities or confidence scores
        
    Returns:
        AUC value as a float
    """
    # Your code here
    if len(set(y_true)) == 1:
        return 0.0
    
    result = []
    thresholds = [float('inf')] + sorted(set(y_scores), reverse=True)
    N = sum([1 for i in y_true if i == 0])
    P = sum([1 for i in y_true if i == 1])
    for threshold in thresholds :
        TP = 0
        FP  = 0
        for i in range(len(y_scores)):
            if y_scores[i] >= threshold :
                if y_true[i] == 1 :
                    TP += 1
                else :
                    FP += 1
        TPR = TP / P
        FPR = FP / N
        result.append((TPR,FPR))
    
    tpr_list, fpr_list = zip(*result)
    auc = 0
    for i in range(1,len(fpr_list)):
        auc += ((tpr_list[i-1] + tpr_list[i]) / 2) * (fpr_list[i] - fpr_list[i-1])
    return auc
    