import numpy as np

def train_bigram_nn(xs, ys, vocab_size, lr=10.0, num_iters=100, alpha=0.0):
	# Initialize W
	W = np.zeros((vocab_size, vocab_size)) 
	# Create empty one-hot matrix
	X = np.zeros((len(xs), vocab_size))
	X[np.arange(len(xs)), xs] = 1
	N = len(xs)
	for _ in range(num_iters):
		# Forward pass
		logits = X @ W
		# probs using softmax
		exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # for stability to avoid overflow
		probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # axis = 1 cuz we need rows not columns 
		# Gradient 
		dz = probs.copy()
		dz[np.arange(N), ys] -= 1
		dz /= N
		# Gradient using W
		dw = X.T @ dz
		# L2 regularization 
		dw += ((2*alpha) / (vocab_size**2))*W
		# update 
		W -= lr*dw

	return W.tolist()
