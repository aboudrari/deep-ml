import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    # Your code here
    X = np.array(X)
    y = np.array(y)
    for i in range(n_epochs):
        if method == 'batch':
            prediction = X @ weights
            error = prediction - y
            gradient = 2 * X.T @ error / len(X)
            weights = weights - learning_rate * gradient
        elif method == 'stochastic':
            for i in range(len(X)):
                X_sample = X[i:i+1]  # shape (1, n)
                y_sample = y[i:i+1]  # shape (1,)
                prediction = X_sample @ weights
                error = prediction - y_sample
                gradient = 2 * X_sample.T @ error 
                weights = weights - learning_rate * gradient
        elif method == 'mini_batch':
            for i in range(0, len(X), batch_size):
                X_batch = X[i : i + batch_size]
                y_batch = y[i : i + batch_size]   
                prediction = X_batch @ weights
                error = prediction - y_batch
                gradient = 2 * X_batch.T @ error / batch_size
                weights = weights - learning_rate * gradient

    return weights        

