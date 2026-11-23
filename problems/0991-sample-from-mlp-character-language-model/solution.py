import numpy as np

def sample_mlp_lm(C, W1, b1, W2, b2, block_size, itos, seed, max_tokens):
	
	# C.shape = (vocab_size, emb_dim)
	C = np.array(C)
	W1 = np.array(W1)
	b1 = np.array(b1)
	W2 = np.array(W2)
	b2 = np.array(b2)
	# zeros array 
	context = np.zeros(block_size, dtype=int)
	rng = np.random.default_rng(seed)
	# Output 
	out = [] 
	for token in range(max_tokens):
		# Embedding lookup
		emb = C[context]
		# Flatten
		flat = emb.flatten()         # or flat = emb.reshape(block_size * emb.shape[1])
		# Hidden layer
		h = np.tanh(flat @ W1 + b1)
		# Output layer || Scores
		logits = h @ W2 + b2
		# Stable Softmax
		maxx = np.max(logits)
		exp_scores = np.exp(logits - maxx)
		probs = exp_scores / np.sum(exp_scores)
		# choice || len(probs) ---> numbers || p=probs ---> proba of each number 
		ix = rng.choice(len(probs), p=probs)

		if ix == 0:
			break

		context = np.append(context[1:], ix)
		# add it to the output
		out.append(itos[ix])
	return "".join(out)