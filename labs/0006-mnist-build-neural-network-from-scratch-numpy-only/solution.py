import numpy as np

class NeuralNetwork:
    '''
    Neural network from scratch using only NumPy.
    Architecture: 784 → 128 (ReLU) → 10 (Softmax)
    '''
    def __init__(self, input_size=784, hidden_size=128, output_size=10, lr=0.01):
        self.lr = lr
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        h1 = np.maximum(z1, 0)                                  # ReLU
        z2 = h1 @ self.W2 + self.b2
        z_shifted = z2 - np.max(z2, axis=1, keepdims=True)      # numerical stability
        exp_z = np.exp(z_shifted)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)    # softmax
        self.cache = {'X': X, 'z1': z1, 'h1': h1, 'z2': z2}
        return probs

    def backward(self, X, y, probs):
        N = X.shape[0]

        # 1. Cross-entropy loss
        correct_probs = probs[np.arange(N), y]
        loss = -np.mean(np.log(correct_probs + 1e-12))

        # 2. Gradient at output (softmax + cross-entropy combined)
        dz2 = probs.copy()
        dz2[np.arange(N), y] -= 1
        dz2 /= N

        # 3. Backprop through layer 2
        h1 = self.cache['h1']
        dW2 = h1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        dh1 = dz2 @ self.W2.T

        # 4. Backprop through ReLU
        dz1 = dh1 * (self.cache['z1'] > 0)

        # 5. Backprop through layer 1
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # 6. Update parameters
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return loss

    def train_step(self, X, y):
        probs = self.forward(X)
        loss = self.backward(X, y, probs)
        return loss

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)