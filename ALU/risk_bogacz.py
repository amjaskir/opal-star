# Lise VANSTEEENKISTE
# internship
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank

# Name: learningbogacz.py

import numpy as np
import random
from bogacz import Bogacz

def simulatebgamble(params,n_states,n_trials,env = "rich", rho=0.0, rnd_seed = None ):


	def calc_D(state):
		"""
		calculates rho for the current trial and returns
		the updated state tracker for rho and respective betas
		"""
		state.D_g[t] = rho
		state.D_n[t] = 1-rho
		return state


	def generate_state():

		if env == "high":
			n_options = 1
			probs = np.random.uniform(.5,1.)
			r_mag = 1
			l_mag = 0
		elif env == "low":
			n_options = 1
			probs = np.random.uniform(0.,.5)
			r_mag = 1
			l_mag = 0
		else:
			err = 'Invalid value given for arg env. %s given' %env
			raise Exception(err)
		new_state = Bogacz(n_trials, n_options, probs, r_mag, l_mag)
		return new_state

	alpha_a, epsilon, K, lbda = params
	states = []

	if rnd_seed is not None:
		random.seed(rnd_seed)
		np.random.seed(rnd_seed)


	for _ in np.arange(n_states):

		state = generate_state()
		for t in range(n_trials):
			state.idx = t
			state = calc_D(state)
			state.policy_gamblesoftmax()
			state.act_gamble(alpha_a, epsilon, lbda)


		states.append(state)

	return states