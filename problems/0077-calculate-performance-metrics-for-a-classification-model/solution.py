import numpy as np 
def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
	# Implement your code here
	actual = np.array(actual)
	predicted = np.array(predicted)
	TP = np.sum((actual == 1) & (predicted ==1))
	TN = np.sum((actual == 0) & (predicted == 0))
	FP = np.sum((actual == 0) & (predicted == 1))
	FN = np.sum((actual == 1) & (predicted == 0))
	confusion_matrix = [[int(TP) , int(FN)],[int(FP) , int(TN)]]
	accuracy = (TP+TN) / (TP+TN+FP+FN)
	specificity = TN / (TN+FP)
	negativePredictive = TN / (TN+FN)
	Precision = TP / (TP+FP)
	Recall = TP / (TP+FN)
	f1 = (2 * (Precision * Recall)) / (Precision + Recall)
    
	return confusion_matrix, round(float(accuracy), 3), round(float(f1), 3), round(float(specificity), 3), round(float(negativePredictive), 3)


