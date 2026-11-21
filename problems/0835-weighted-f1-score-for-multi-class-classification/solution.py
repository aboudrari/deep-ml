import numpy as np

def weighted_f1_score(y_true, y_pred) -> float:
    
    total_F1 = [] 
    union = set(y_true) | set(y_pred)  

    for n in union :
        TP = sum(1 for t, p in zip(y_true, y_pred) if (t == n and p == n))
        FP = sum(1 for t, p in zip(y_true, y_pred) if (t != n and p == n))
        FN = sum(1 for t, p in zip(y_true, y_pred) if (t == n and p != n))
        support = sum(1 for t in y_true if t == n)
        if TP + FP == 0 :
             Precision = 0 
        else :
            Precision = TP / (TP + FP)
        if TP + FN == 0 :
            Recall = 0 
        else :
            Recall = TP / (TP + FN)

        if Precision + Recall == 0 :
            f1 = 0 
        else :
            f1 = F_1 = 2 * ((Precision * Recall) / (Precision + Recall))
            
        total_F1.append(f1 * support)
        
    return round(sum(total_F1) / len(y_true), 4)