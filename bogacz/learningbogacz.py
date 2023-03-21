# Lise VANSTEEENKISTE
# internship
#
# Edits by Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank

# Name: learningbogacz.py

import numpy as np
import random
from bogacz import Bogacz
from environments import calc_probs

def simulate(params,n_states,n_trials,env = "rich", policy="softmax",\
 D=0.5, mod = "constant",thresh = 0, k=1,rnd_seeds = None, V0=0.0, full=False,
 rmag = 1, lmag = 0):
	""" Simulate bogacz in learning environment.

	Inputs:
	- n_states, int
	- n_trials, int
	- env, "rich" or "lean"
	- policy, "policy_max" or "softmax"
	- D, D value for all of learning
	- modulation
	- thresh, trial after which D is set from 1/2 to specified D
	- k, for value modulation, degree value influences DA
	- rnd_seed, set random seed of stim if specified
	"""

	def calc_D(state):
		"""
		calculates D for the current trial and returns
		the updated state tracker for D and respective betas

		D represents dopamine levels (equivalent of rho in OpAL)
		Scales between 0 and 1, with 1 high level of DA
		"""
		if t < thresh:
			state.D_g[t] = 0.5
			state.D_n[t] = 0.5
		else:
			if mod == "constant":
				state.D_g[t] = D
				state.D_n[t] = 1-D
			if mod == "value":
				# NOTE: if rmag and lmag is 1/0, can just use V
				# average of two actions
				V = np.mean(1/2*(state.QG[t,:] - state.QN[t,:])) # state average(?)  
				V = 1/(1 + np.exp(-V*k))  # translate between 0 and 1
				state.D_g[t] = V 
				state.D_n[t] = 1 - V
		return state


	def generate_state():
		"""
		Get appropriate reward probabilities and magnitudes
		for the specified environment type
		"""

		probs = calc_probs(env)
		n_options = len(probs)

		# feedback for agent
		r_mag = np.zeros(n_options) + rmag
		l_mag = np.zeros(n_options) + lmag

		new_state = Bogacz(n_trials, n_options, probs, r_mag, l_mag, V0=V0)
		return new_state


	# learning rate, damping, decay, softmax temp
	alpha_a, epsilon, lbda, beta = params
	states = []

	# do the thing
	for s in np.arange(n_states):

		# check if random seed provided
		if rnd_seeds is not None:
			random.seed(rnd_seeds[s])
			np.random.seed(rnd_seeds[s])

		state = generate_state()
		for t in range(n_trials):

			state.idx = t
			state=calc_D(state)					# get D
			state.policy_softmax(beta)
			state.act(alpha_a, epsilon, lbda)	# update 

			if full:
				state.update_other_actions(alpha_a, epsilon, lbda)

		states.append(state)					# save sim

	return states



