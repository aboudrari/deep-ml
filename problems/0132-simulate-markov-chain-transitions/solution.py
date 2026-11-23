import numpy as np
def simulate_markov_chain(transition_matrix, initial_state, num_steps):
	# Your code here
	current_state = initial_state

	states = [current_state]

	for step in range(num_steps):

		probabilities = transition_matrix[current_state]

		next_state = np.random.choice(transition_matrix.shape[0],p=probabilities)

		states.append(next_state)

		current_state = next_state

	return states
