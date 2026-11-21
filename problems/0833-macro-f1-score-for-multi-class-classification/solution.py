def macro_f1(y_true: list, y_pred: list) -> float:
    """
    Compute the Macro F1 score over all classes appearing in y_true or y_pred.
    Returns a single float.
    """
    total_F1 = []
    # input : y_true = [0, 0, 1, 1, 2, 2], y_pred = [0, 1, 1, 1, 2, 0]
    union = set(y_true) | set(y_pred)
    for c in union:
        TP = sum(1 for t, p in zip(y_true, y_pred) if (t == c and p == c))
        FP = sum(1 for t, p in zip(y_true, y_pred) if (t != c and p == c))
        FN = sum(1 for t, p in zip(y_true, y_pred) if (t == c and p != c))
        if TP + FP == 0 :
            Precision = 0
        else :
            Precision = TP / (TP + FP)
        if TP + FN == 0 :
            Recall = 0
        else :
            Recall = TP / (TP + FN)
        if Precision + Recall == 0 :
            F_1 = 0 
        else :
            F_1 = 2 * ((Precision * Recall) / (Precision + Recall))
        total_F1.append(F_1)
    macro_average = sum(total_F1) / len(union)
    return macro_average 
