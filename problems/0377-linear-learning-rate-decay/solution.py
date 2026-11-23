def linear_lr_decay(initial_lr: float, end_lr: float, num_steps: int) -> list:
	# If num_steps is 0, return an empty list.
	# If num_steps is 1, return a list containing only initial_lr

	if num_steps == 0 :
		return []
	if num_steps == 1 :
		return [initial_lr]

	learning_rates = []
	for step in range(num_steps): # lr = 0 
		lr = initial_lr + (end_lr - initial_lr) * (step / (num_steps - 1)) # 0.1 + (0.0 - 0.1)*(0/4)
		learning_rates.append(lr)
	return learning_rates