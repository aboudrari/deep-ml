import numpy as np

def micro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))

    if TP + FP != 0:
        P_micro = TP / (TP + FP)
    else:
        P_micro = 0

    if TP + FN != 0:
        R_micro = TP / (TP + FN)
    else:
        R_micro = 0

    if P_micro + R_micro != 0:
        F1_micro = 2 * (P_micro * R_micro) / (P_micro + R_micro)
    else:
        F1_micro = 0

    return F1_micro