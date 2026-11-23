import numpy as np

def compute_tpr_fpr(y_true, y_pred):
	
	TP = np.sum((y_true == 1) & (y_pred == 1))
	TN = np.sum((y_true == 0) & (y_pred == 0))
	# ✅ Correct
    FN = np.sum((y_true == 1) & (y_pred == 0))   # real positive, missed it
    FP = np.sum((y_true == 0) & (y_pred == 1))   # real negative, wrongly flagged

	if TP + FN == 0 :
		TPR = 0.0
	else :
		TPR = TP / (TP + FN)

	if FP + TN == 0 :
		FPR = 0.0 
	else :
		FPR = FP / (FP + TN)
	return (TPR,FPR)

