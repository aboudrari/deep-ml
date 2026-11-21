from collections import Counter

def hard_voting_classifier(predictions: list[list[int]]) -> list[int]:
    """
    Implement a hard voting classifier using majority vote.
    
    Args:
        predictions: 2D list where predictions[i][j] is classifier i's prediction for sample j
        
    Returns:
        List of final predictions using majority vote
    """
    #predictions = [[0, 1, 2], 
	#               [0, 2, 2], 
	#               [1, 1, 2]]
    winners = []
    for i in range(len(predictions[0])) :
        votes = [predictions[j][i] for j in range(len(predictions))]
        c = Counter(votes)
        max_count = max(c.values())
        tied_classes = [cls for cls in c if c[cls] == max_count]
        winner = min(tied_classes)
        winners.append(winner)
    return winners

