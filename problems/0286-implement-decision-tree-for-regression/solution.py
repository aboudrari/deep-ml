import numpy as np

def decision_tree_regressor(X_train, y_train, X_test, max_depth=2, min_samples_split=2):
    
    def mse(y): 
        MSE = np.mean((y - np.mean(y))**2)
        return MSE
    
    def find_best_split(X, y):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        for i in range(X.shape[1]):
              unique_vals = np.unique(X[:, i])
              thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2 
              for threshold in thresholds :
                  left_mask = X[:, i] <= threshold
                  right_mask = X[:, i] > threshold
                  y_left = y[left_mask]
                  y_right = y[right_mask]
                  if len(y_left) == 0 or len(y_right) == 0:
                      continue 
                  weighted_MSE = (len(y_left)*mse(y_left) + len(y_right)*mse(y_right)) / (len(y_right)+len(y_left))
                  Gain = mse(y) - weighted_MSE
                  if Gain > best_gain:
                      best_gain = Gain
                      best_feature = i
                      best_threshold = threshold
        return best_feature, best_threshold
                    
    def build_tree(X, y, depth): 
        if depth >= max_depth or len(X)< min_samples_split or len(np.unique(y)) == 1:
            return {'leaf': True, 'value': np.mean(y)}
        feature, threshold = find_best_split(X, y) 
        if feature is None: 
            return {'leaf': True, 'value': np.mean(y)}
        
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold
        y_left = y[left_mask]
        y_right = y[right_mask]
        X_left = X[left_mask]
        X_right = X[right_mask]
        left_subtree = build_tree(X_left, y_left, depth + 1)
        right_subtree = build_tree(X_right, y_right, depth + 1)
        return {'leaf': False, 'feature': feature, 'threshold': threshold, 'left': left_subtree, 'right': right_subtree}
    
    def predict_sample(node, x) :
        if node['leaf'] == True :
            return node['value']
        elif x[node['feature']] <= node['threshold']  :
            return predict_sample(node['left'], x)
        else :
            return predict_sample(node['right'], x)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    tree = build_tree(X_train, y_train, 0)
    predictions = []
    for sample in X_test :
        predictions.append(predict_sample(tree, sample))
    return np.round(predictions, 4).tolist()
