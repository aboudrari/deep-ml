import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs):
    features = np.array(features)
    labels = np.array(labels)
    w = np.array(initial_weights, dtype=float).copy()
    b = initial_bias
    mse_values = []

    for epoch in range(epochs):
        z = np.dot(features, w) + b
        y_hat = sigmoid(z)
        MSE = np.mean((y_hat - labels)**2)
        delta = (y_hat - labels) * y_hat * (1 - y_hat)
        dL_dw = 2 * np.dot(features.T, delta) / len(features)
        dL_db = 2 * np.mean(delta)

        w = w - learning_rate * dL_dw
        b = b - learning_rate * dL_db

        mse_values.append(round(MSE, 4))

    return np.round(w, 4), round(b, 4), mse_values