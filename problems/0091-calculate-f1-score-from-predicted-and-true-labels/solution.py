def calculate_f1_score(y_true, y_pred):
	"""
	Calculate the F1 score based on true and predicted labels.

	Args:
		y_true (list): True labels (ground truth).
		y_pred (list): Predicted labels.

	Returns:
		float: The F1 score rounded to three decimal places.
	"""
	# Your code here
	TP = sum( 1 for i , j in zip(y_true , y_pred) if i == 1 and j == 1)
	TN = sum( 1 for i , j in zip(y_true , y_pred) if i == 0 and j == 0 )
	FP = sum( 1 for i , j in zip(y_true , y_pred) if i == 0 and j == 1 )
	FN = sum( 1 for i , j in zip(y_true , y_pred) if i == 1 and j == 0 )
	
	Precision = 0 if (TP + FP) == 0 else TP / (TP + FP)
	Recall = 0 if (TP + FN) == 0 else TP / (TP + FN)

	f1 =  0 if (Precision + Recall) == 0 else 2 * ((Precision * Recall) / (Precision + Recall))
	return round(f1,3)

