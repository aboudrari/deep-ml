def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
	# Return the element-wise sum of vectors 'a' and 'b'.
	# If vectors have different lengths, return -1.
	q , p = len(a) , len(b)
	if q != p :
		return -1
	else :
		return [a + b for a , b in zip(a,b)] 