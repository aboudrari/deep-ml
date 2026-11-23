# Using Character-level tokenizer

def train_tokenizer(corpus, vocab_size):
	
	chars = sorted(list(set("".join(corpus))))
	
	# mappings
	# string to int 
	stoi = {s:i for i, s in enumerate(chars)}  
	# int to str 
	itos = {i:s for i,s in enumerate(chars)}
	# encode 
	def encode(text):
		return [stoi[chr] for chr in text]

	# decode 
	def decode(tokens):
		return ''.join(itos[t] for t in tokens)

	return encode, decode