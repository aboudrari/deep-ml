import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.
    epochs = 500  
    n = len(X)
    learning_rate = 0.1
    for epoch in range(epochs):
        y_pred = X @ W + b
        errors = y_pred - y
        grad_W = (2/n) * X.T @ errors
        grad_b = (2/n) * np.sum(errors)

        W = W - learning_rate * grad_W
        b = b - learning_rate * grad_b
    return W, b


    
