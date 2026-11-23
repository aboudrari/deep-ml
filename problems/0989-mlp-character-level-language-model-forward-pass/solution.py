import numpy as np

def mlp_forward_loss(X, Y, C, W1, b1, W2, b2):
	"""
	Forward pass of an MLP character-level language model.
	Returns mean cross-entropy loss as a float.
	"""
	# X.shape = (N, block_size)
	# Y.shape = (N,)
	# C.shape = (Vocab_size, emb_dim)

	emb = C[X]        # emb.shape = (N, block_size, emb_dim)
	N, block_size, emb_dim = emb.shape

    # flatten
	flat  = emb.reshape(-1, block_size * emb_dim)    # flat.shape = (N, block_size * emb_dim)

	# W1.shape = (block_size * emb_dim, hidden) 
	h = np.tanh(flat @ W1 + b1)   # flat.shape = (32, 30) and W1.shape = (30, 32) --> (flat @ W1).shape = (32, 32) + b1.shape = (32,) = (32, 32)

	# Computes output logits:h.shape = (N, hidden) || W2.shape = (hidden, vocab_size) || b2.shape = (vocab_size,) || logits.shape = (N, vocab_size)
	logits = h @ W2 + b2  # scores 
	maxx = np.max(logits, axis=1, keepdims=True)
	exp_scores = np.exp(logits - maxx)    # exp_scores.shape = (32,27)
	denom = np.sum(exp_scores, axis=1, keepdims=True)
	probs = exp_scores / denom
	# print(probs.sum(axis=1))   # each row should be = 1
	
	# Cross Entropy Loss 
	correct_probs = probs[range(N), Y]  # advanced indexing
	# likelihood 
	loss = -np.log(correct_probs).mean()
	return loss
