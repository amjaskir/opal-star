# Lise VANSTEEENKISTE
# internship
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank

# Name: dynamics_bogacz.py

import numpy as np
import random
from bogacz import Bogacz



def simulate_bogacz(n_trials, alpha_a, epsilon, lbda, full = True):
	# initialize parameters

	state = Bogacz(n_trials,1,[.5],[20],[-20])

	for t in range(n_trials):

		if np.mod(t,2) == 0:
			state.probs = [0]
		else:
			state.probs = [1]
		state.policy_forced(0)

		if full:	# use full model with decay
			state.act(alpha_a, epsilon, lbda)
		else:
			state.act_simple(alpha_a, epsilon, lbda)
		state.idx = state.idx + 1

	return state

