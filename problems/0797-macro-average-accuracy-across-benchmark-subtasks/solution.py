def macro_average_accuracy(subtask_results: dict) -> float:
    """
    Compute the unweighted macro-average accuracy across benchmark subtasks.
    Each value in subtask_results is a list of (prediction, label) tuples.
    Return the macro-average accuracy rounded to 4 decimals.
    """
    # example : subtask_results = {"a": [(1,1),(0,1)], "b": [(1,1),(1,1),(0,0),(1,0)]}
    # if the dict is emptey return 0.0
    if not subtask_results :
        return  0.0 
    # empty list to store accuracy 
    accuracys = []
    # here we need only the values  
    for i in subtask_results.values():
        # you need to check if i = [(1,1),(0,1)] then i = [(1,1),(1,1),(0,0),(1,0)]
        if not i :
            accuracys.append((0.0))
            continue
        else :
            n = len(i)
            # here you need to check if we have  correct prediction like {for num in i  ---> i = (1,1)} they are the same and then i = (0,1) 
            correct = sum(1 for num in i if num[0] == num[1] )
            accuracy = correct / n
            accuracys.append(accuracy)
    Macro_average = sum(accuracys) / len(accuracys)
    return round(Macro_average,4)