import  numpy as np 
def dice_score(y_true, y_pred):
	# Write your code here
	TP = sum ((y_true == 1) & (y_pred == 1))
	TN = sum((y_true == 0) & (y_pred == 0))
	FN = sum( (y_true == 1) & (y_pred == 0) )
	FP = sum((y_true == 0) & (y_pred == 1))
	res = 0 if (2*TP + FP + FN) == 0 else (2 * TP) / (2*TP + FP + FN)
	return round(res, 3)

