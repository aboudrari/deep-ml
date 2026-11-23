import numpy as np

def sample_name(P, itos, seed, max_len=100):
	# Your code here
	rng = np.random.default_rng(seed) 
	ix = 0
	out = []
	probs = []

	for row in P:
		sum_ = np.sum(row)

		if sum_ == 0:
			zero_row = np.zeros(P.shape[0])
			zero_row[0] = 1
			probs.append(zero_row)
		else:
			probs.append(row / sum_)

	probs = np.array(probs)

	for _ in range(max_len):
		row = probs[ix] 
		r = rng.random()
		cdf = np.cumsum(row)
		j = np.searchsorted(cdf, r)

		if j == 0:
			break 
		else :
			out.append(itos[j])
			ix = j
	return ''.join(out)

