# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
import numpy as np
import rl
import random
from rl import RL
from environments import get_probs

import time

def simulate(params,n_states,n_trials,v0=0.0,env = "rich",\
	r_mag = 1, l_mag = 0, rnd_seeds = None, gamble = False,\
	UCB = False, UCB_samplemean=False, gamma = 0,
	forced_actions = None, forced_rewards = None):
	"""
	Simulates learning in specified environment, each a multiarm
	bandit where success receives +1 and failure receives -1, for 
	specified number of trials. Can modulate DA at choice (rho) online
	by critic value or set to specified constant value 

	Inputs:
	params - tuple of (alpha, beta)
			 - alpha, learning rate 
			 - beta, softmax inverse temp for policy
	n_states - number of states of learning to simulate
	n_trials - number of trials of learning within each state
	v0 - critic initialization value
	rnd_seed - sets random seed generator to desired seed at beginning of 
			   simulations
	full 	- model gets full information
	c_only 	- choice only updates critic if full is True


	Outputs:
	states - contains tracker for each simulated learning state
		   - see rl.py for more detail
	"""

	##########################################################################

	def generate_state(params):
		"""
		intialize state for learning according to specified probs
		assumes same r_mag and l_mag for each option
		"""
		probs = get_probs(env)
		n_options = len(probs)
		new_state = RL(n_trials, v0, n_options, probs, r_mag, l_mag, params, gamble, \
			UCB=UCB, UCB_samplemean=UCB_samplemean, gamma=gamma)
		return new_state
	##########################################################################

	start = time.time()
	states = []

	# let's do this thing
	for s in np.arange(n_states):

		# check if random seed provided
		if rnd_seeds is not None:
			random.seed(rnd_seeds[s])
			np.random.seed(rnd_seeds[s])

		# generate new learning state
		# print("Starting new state", s, time.time() - start)
		state = generate_state(params)
		for t in range(n_trials):
			state.idx = t
			state.policy(forced_actions=forced_actions,forced_rewards=forced_rewards,state_idx=s)	# pick an action and generate PE
			state.update()	# update Q value with PE

		# save state learning
		states.append(state)

	return states