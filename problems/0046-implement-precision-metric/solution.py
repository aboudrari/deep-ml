import numpy as np
def precision(y_true, y_pred):
	# Your code here
	TP = np.sum((y_true==1) & (y_pred==1))
	FP = np.sum((y_true==0) & (y_pred==1))
	Precision = TP / (TP + FP)
	return Precision
