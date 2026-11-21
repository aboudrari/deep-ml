import numpy as np

def accuracy_score(y_true, y_pred):
	# Your code here
	# Accuracy = (TP + TN) / (TP + TN + FP + FN)
	n = len(y_true)
	count = 0
	for index , value in enumerate(y_true):
		if value == y_pred[index]:
			count += 1
			
	accuracy = count / n 
	return accuracy	

		