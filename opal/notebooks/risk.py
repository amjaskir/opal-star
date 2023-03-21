# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
import numpy as np
import random
from scipy.stats import beta as beta_rv

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# my modules
from opal import OpAL

def simulate(params,n_states,n_trials,v0=0.0,crit="S",env = "rich", \
	mod = "zero", k=1., rho=0.0, r_mag = 1, l_mag = -1, rnd_seed = None):
	"""
	Simulates decision making in gambles in states of either high (>50%)
	or low (<50%) gamble probabilities for specified number of trials. 
	Can modulate DA at choice (rho) online by critic value or set to 
	specified constant value 

	Inputs:
	params - tuple of (alphaC, alphaA, beta)
			 - alphaC, learning rate of the critic
			 - alphaA, learning rate of the actor
			 - beta, softmax inverse temp for policy
	n_states - number of states of learning to simulate
	n_trials - number of trials of learning within each state
	v0 - critic initialization value
	crit - "S" or "SA"
		   - critic tracks overall state value
		   - critic tracks state-action pairs
	mod - "constant", rho set to constant value throughout 
		- "value", rho modulated online by learned critic value
				   max critic value if critic is tracking state-action pairs
		- "avg_value", if critic tracks state-action values, modulates rho
					   by average over all values
	k - multiplier for rho modulation
	rho - if mod == "constant", desired rho level
	rnd_seed - sets random seed generator to desired seed at beginning of 
			   simulations

	Outputs:
	states - contains tracker for each simulated learning state
		   - see classes.py for more detail
	"""

	##########################################################################
	def calc_rho(state):
		"""
		calculates rho for the current trial and returns
		the updated state tracker for rho and respective betas
		"""
		if mod == "constant": 		# artificially set rho to specific value, e.g. rho = 0
			state.rho[t] = rho
		elif mod == "value":		# state's critic value, max if crit = SA
			state.rho[t] = np.max(state.V[t])*k
		elif mod == "beta":
				alph = state.V[0]
				bet = state.V[1]
				mean, var = beta_rv.stats(alph,bet,moments='mv')
				state.mean = mean
				# assume rmag and lmag are same for both options
				infer_val = r_mag*mean + l_mag*(1-mean) # in [-1,1], like rho
				state.rho[t] = infer_val*k 
		else:
			err = 'Invalid value given for arg mod. \"%s\" given' %mod
			raise Exception(err)

		state.beta_g[t] = np.max([0,beta*(1 + state.rho[t])])
		state.beta_n[t] = np.max([0,beta*(1 - state.rho[t])])
		return state

	def generate_state():
		"""
		intialize state for learning according to specified environment type

		r_mag = 1  # get double s.t.
		l_mag = -1 # lose s.t.
		"""
		if env == "high":
			n_options = 1
			probs = np.random.uniform(.5,1.)
		elif env == "low":	
			n_options = 1							
			probs = np.random.uniform(0.,.5)
		else:
			err = 'Invalid value given for arg env. %s given' %env
			raise Exception(err)
		new_state = OpAL(n_trials, crit, v0, n_options, probs, r_mag, l_mag)
		return new_state
	##########################################################################

	# define parameters
	alpha_c, alpha_a, beta = params
	states = []

	# check if random seed provided
	if rnd_seed is not None:
		random.seed(rnd_seed)
		np.random.seed(rnd_seed)

	# let's do this thing
	for s in np.arange(n_states):

		# generate new learning state
		state = generate_state()
		for t in range(n_trials):
			state.idx = t
			state = calc_rho(state)	    # calculate rho, beta_g, beta_n
			state.policy_gamble()		# pick an action and generate PE
			state.critic(alpha_c)		# update critic with PE
			state.act_gamble(alpha_a)	# update actors with PE

		# save state learning
		states.append(state)

	return states