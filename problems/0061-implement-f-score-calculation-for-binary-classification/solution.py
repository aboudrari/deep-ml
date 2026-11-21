import numpy as np

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	TP = np.sum((y_true == 1 ) & (y_pred == 1))
	TN = np.sum((y_true == 0 ) & (y_pred == 0))
	FN = np.sum((y_true == 1 ) & (y_pred == 0))
	FP = np.sum((y_true == 0 ) & (y_pred == 1))
	
	Recall = TP / (TP + FN )
	Precision = TP / (TP + FP)
	if (beta**2 * Precision) + Recall == 0:
		return 0.0
	F_Score = ( 1 + beta**2) * ((Precision * Recall) / ((beta**2 * Precision) + Recall) )
	return np.round(F_Score , 3)
