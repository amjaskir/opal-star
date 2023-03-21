# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
import numpy as np
import random

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# my modules
from opal import OpAL
from opal import TrialTracker

def simulate(params,n_states,n_trials,v0=0.0,crit="SA",env = "rich", \
	mod = "constant", k=8., rho=0.0, rnd_seed = None, high_p = .5, set_prior = False):
	"""
	Simulates learning in reward gamble environments = [.75,.25] for 
	specified number of trials. Can modulate DA at choice (rho) online
	by critic value or by infering whether the current learning
	state is either rich or lean

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
		- "infer", rho set to P(rich)*EV(rich) + P(lean)*EV(lean)
				   given reward history using Bayes rule
		- "value", rho modulated online by learned critic value
				   max critic value if critic is tracking state-action pairs
		- "avg_value", if critic tracks state-action values, modulates rho
					   by average over all values
	k - multiplier for rho modulation
	rho - if mod == "constant", desired rho level
	rnd_seed - sets random seed generator to desired seed at beginning of 
			   simulations
	high_p - proportion of high gambles vs low gambles
	set_prior - if many states, proportion of high gamble states as prior

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
		elif mod == "infer":		# P(rich)*EV(rich) + P(lean)*EV(lean)
			# average expected values for each state type
			if env == "bimodal":
				EVs = np.array([(.8*1 + .2*-1), (.2*1+.8*-1)])	
			else:
				EVs = np.array([(.8*1 + .2*-1), (.6*1+.4*-1)])	
			state.rho[t] = np.dot(state.prior[t,:],EVs)*k
		elif mod == "value":		# state's critic value, max if crit = SA
			state.rho[t] = np.max(state.V[t])*k
		elif mod == "avg_value":	# avg state critic value (if crit = SA)
			state.rho[t] = np.avg(state.V[t])*k
		else:
			err = 'Invalid value given for arg mod. \"%s\" given' %mod
			raise Exception(err)

		#state.beta_g[t] = beta*(1 + state.rho[t])
		#state.beta_n[t] = beta*(1 - state.rho[t])

		state.beta_g[t] = np.max([0,beta*(1 + state.rho[t])])
		state.beta_n[t] = np.max([0,beta*(1 - state.rho[t])])
		return state

	def generate_state():
		"""
		intialize state for learning according to specified environment type
		"""
		if env == "uniform":
			n_options = 1
			# mean probability, uniform from .15 to .85
			p = np.random.uniform(.15,.85)
			probs = np.array([p])
			r_mag = np.array([1])
			l_mag = np.array([-1])
		elif env == "bimodal":
			n_options = 1
			pick = np.random.binomial(size=1, n=1, p=high_p)[0]
			state_mean = pick*.6 + .2 	# .8 or .2
			probs = np.array([np.random.uniform(state_mean-.05,state_mean+.05)])
			r_mag = np.array([1])
			l_mag = np.array([-1])
		elif env == "bimodal_high":
			n_options = 1
			pick = np.random.binomial(size=1, n=1, p=high_p)[0]
			state_mean = .6 + .2*pick #.6 or .8
			probs = np.array([np.random.uniform(state_mean-.05,state_mean+.05)])
			r_mag = np.array([1])
			l_mag = np.array([-1])
		else:
			err = 'Invalid value given for arg env. %s given' %env
			raise Exception(err)
		new_state = OpAL(n_trials, crit, v0, n_options, probs, r_mag, l_mag)
		
		# set prior is specified
		if set_prior:
			if tracker.idx > 0:
				col_sum = np.nansum(tracker.state_priors, axis=0)
				state.prior[0,:] = col_sum/np.sum(col_sum)

		return new_state

	def calcPosterior(state):
		""" calculates posterior evidence for being in a high or low gamble 
		given the reward (or no reward) received ON A GAMBLE
		"""

		# # carry over my last values
		# state.prior[t+1,:] = state.prior[t,:]

		if state.C[t] == 1:
			# I took the gamble, so update my beliefs with the new info
			reward = state.R[t]
			priors = state.prior[t,:]
			if reward == 1:
				if env == "bimodal": 
					likelihoods = np.array([.8, .2])	# l(reward|state type)
				else:
					likelihoods = np.array([.8, .6])	# l(reward|state type)
			else:
				if env == "bimodal":
					likelihoods = 1 - np.array([.8, .2])# l(no reward|state type)
				else:
					likelihoods = 1 - np.array([.8, .6])# l(no reward|state type)s
			evidence = np.dot(priors,likelihoods)		# marginal likelihood of (no) reward
			posterior = (priors*likelihoods)/evidence
			state.prior[t+1,:] = posterior
		else:
			# carry over my last values
			state.prior[t+1,:] = state.prior[t,:]
		return state
	##########################################################################

	# define parameters
	alpha_c, alpha_a, beta = params
	states = []

	# check if random seed provided
	if rnd_seed is not None:
		random.seed(rnd_seed)
		np.random.seed(rnd_seed)

	tracker = TrialTracker(n_states)			# internal tracks state priors
	# let's do this thing
	for s in np.arange(n_states):

		# generate new learning state
		state = generate_state()
		for t in range(n_trials):
			state.idx = t
			state = calc_rho(state)	    	# calculate rho, beta_g, beta_n
			state.policy_gamble()			# pick an action and generate PE
			state.critic(alpha_c)			# update critic with PE
			state.act_gamble(alpha_a)		# update actors with PE
			state = calcPosterior(state) 	# update state type belief

		# save state learning
		states.append(state)

		# update tracker if needed
		if set_prior:
			tracker.state_priors[tracker.idx,:] = state.prior[n_trials,:]
			tracker.idx = tracker.idx + 1

	return states