import numpy as np

def precision_recall_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute precision-recall pairs for different probability thresholds.
    
    Args:
        y_true: List of true binary labels (0 or 1)
        y_scores: List of predicted probabilities or confidence scores
    
    Returns:
        Tuple of (precisions, recalls, thresholds) where each is a list
    """
    # Your code here
    precisions  = []
    recalls = []
    total_positives = sum(y_true)
    thresholds = sorted(set(y_scores), reverse=True)
    for threshold in thresholds:
      TP = 0
      FP = 0
      for score, label in zip(y_scores, y_true):
        if score >= threshold:
            if label == 1:
               TP += 1
            else:
               FP += 1
      if TP + FP == 0:
         precision = 1
      else:
         precision = TP / (TP + FP)
      precisions.append(precision)
      if total_positives == 0:
         recall = 0.0
      else:
         recall = TP / total_positives
      recalls.append(recall)
    
    return precisions, recalls, thresholds  # ← outside the loop