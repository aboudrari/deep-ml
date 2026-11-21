
def confusion_matrix(data):
    tp, fn, fp, tn = 0, 0, 0, 0
    for pair in data:
        y_true, y_pred = pair[0], pair[1]
        # your if/elif logic here
        if y_true == 0 and y_pred == 0 :
            tn +=1
        elif y_true == 1 and y_pred == 1 :   
            tp += 1
        elif y_true == 0 and y_pred == 1 :
            fp += 1
        else :
            fn += 1        
    return [[tp , fn] , [fp , tn ]]



