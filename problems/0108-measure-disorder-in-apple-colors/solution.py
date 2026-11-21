from collections import Counter
# Counter([1, 1, 0, 0])
# gives: {1: 2, 0: 2}
def disorder(apples: list) -> float:
	"""
	Compute the disorder in a basket of apples.
	"""
	# Your code here
	apple = Counter(apples)
	total = len(apples)
	gini  = 0
	for count in apple.values():
		p = count / total
		gini += p **2
	return 1 - gini

