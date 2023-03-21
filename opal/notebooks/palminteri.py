# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
import numpy as np
import opal
import random
from scipy.stats import beta as beta_rv
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from opal import OpAL
from environments import get_probs
import itertools

import time

def simulate(params,n_states,v0=0.0,crit="S",\
	mod = "constant", k=8., phi = 3,
	rho=0.0, baselinerho = 0.0, threshRho = 0, threshRand = 0, 
	rnd_seeds = None,\
	norm=False,mag=1,norm_type = None,\
	hebb=True,\
	anneal = False, T = 100.0,use_var=False,\
	decay_to_prior = False, decay_to_prior_gamma = 1.,
	pgrad=False):
	"""
	Simulates learning in specified environment, each a multiarm
	bandit where success receives +1 and failure receives -1, for 
	specified number of trials. Can modulate DA at choice (rho) online
	by critic value or set to specified constant value 

	Inputs:
	params - tuple of (alphaC, alphaA, beta)
			 - alphaC, learning rate of the critic
			 - alphaA, learning rate of the actor
			 - beta, softmax inverse temp for policy
	n_states - number of states of learning to simulate
	n_trials - number of trials of learning within each state
	v0 - critic initialization value
	crit - "S" or "SA"
		   - "S"  critic tracks overall state value
		   - "SA" critic tracks state-action pairs
	env - specified environment with reward probabilities of options
		  see environments.py for options
	mod - "constant", rho set to constant value throughout 
		- "value", rho modulated online by learned critic value
				   max critic value if critic is tracking state-action pairs
		- "avg_value", if critic tracks state-action values, modulates rho
					   by average over all values

	k - multiplier for rho modulation
	rho - if mod == "constant", desired rho level
	rnd_seed - sets random seed generator to desired seed at beginning of 
			   simulations
	norm - T/F, normalize actor learning PE between [0,1] according to specified mag
	mag - scalar, unsigned magnitude of largest feedback for normalization
	hebb - T/F, use three-hactor hebbian term
	variant - str, specifies OpAL variant, see code for description
				"flip", "bmod","lrate"
	full 	- model gets full information
	phi - param for # of std above/below .5 to modulate rho

	switch - introduce switch points?
	env_switch - environment post switch point


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

		# use base rho for earlier trials
		if t < threshRho: 
			state.rho[t] = rho
		else:
			chance = state.r_mag[0]*.5 + state.l_mag[0]*.5 
			if mod == "constant": 		# artificially set rho to specific value, e.g. rho = 0
				state.rho[t] = rho + baselinerho
			elif mod == "value":		# modulate rho by state's critic value, max if crit = SA 	NOTE TO SELF: In lean environments, this is slightly inaccurate - opal maybe choosing the correct option, but the max may still be the other option. In this case, average is more appropriate 
				state.rho[t] = np.max(state.V[t] - chance)*k
			elif mod == "avg_value":	# avg state critic value (if crit = SA)
				state.rho[t] = np.mean(state.V[t] - chance)*k
			elif mod == "avg_gamble_value":
				if t == 0:
					state.rho[t] = 0
				else:
					uniform_dist = 0.5*((1+4/2)) + 0.5*(0.5*(2+8)/2 + 0.5*0) # EV if X, C were uniform distr (flat prior)
					state.rho[t] = k*(np.mean(.5*state.Cs[0:t] + .5*(0.5*state.Xs[0:t] +  0.5*0)) - uniform_dist) # EV of gambles til now relative to flat prior
			elif mod == "beta":
				# get the mean val of the environment
				# calculated in opal.py during critic update
				mean = state.mean[t]

				# assume rmag and lmag are same for both options
				infer_val = state.r_mag[0]*mean + state.l_mag[0]*(1-mean) # in [0,1], like rho
				chance = state.r_mag[0]*.5 + state.l_mag[0]*.5 # value that dictates chance
				
				# am I sufficiently above/below 50%?
				# lb = mean - phi*state.std[t]
				# ub = mean + phi*state.std[t]
				# ranges = np.array([lb,ub])
				# check_me = sum(ranges < .5)
				# sufficient = ((check_me == 2) or (check_me == 0))
				thresh = phi*state.std[t]
				cond1 = (mean - thresh) > .5 # lower bound above .5
				cond2 = (mean + thresh) < .5 # upper bound below .5
				sufficient = cond1 or cond2

				# is sufficiently above/below .5 or 
				# always modify based on mean
				if sufficient:
					# calc rho val direction
					# above p(reward) = .50 is positive, below negative
					# use mean of state
					chance_centered_val = (state.mean[t] - .5)
					state.rho[t] = chance_centered_val*k + baselinerho
				else:
					state.rho[t] = rho + baselinerho
			else:
				err = 'Invalid value given for arg mod. \"%s\" given' %mod
				raise Exception(err)

		this_r = state.rho[t]
		state.beta_g[t] = np.max([0,beta*(1 + this_r)])
		state.beta_n[t] = np.max([0,beta*(1 - this_r)])

		return state

	def generate_state(n_trials):
		"""
		intialize state for learning according to specified probs
		assumes same r_mag and l_mag for each option
		"""

		n_options = 8
		probs = np.array([.75,.25,.75,.25,.75,.25,.75,.25])
		rmag = 	np.array([.5,.5,.5,.5,.0,.0,.0,.0])
		lmag = 	np.array([.0,.0,.0,.0,-.5,-.5,-.5,-.5])
		ctx = np.array([0,0,1,1,2,2,3,3])

		full_info = [True,True,False,False,True,True,False,False]	# full or partial info
		order = np.repeat([0,1,2,3],24)								# order of ctx types
		np.random.shuffle(order)

		new_state = OpAL(n_trials,crit,v0,n_options,probs,rmag,lmag,\
							anneal=anneal,use_var=use_var,T=T,norm_type=norm_type, pgrad=pgrad)
		new_state.ctx = ctx
		new_state.ctx_order = order
		new_state.full_info = full_info

		return new_state
	
	##########################################################################
	
	# define parameters
	alpha_c, alpha_a, beta = params
	n_trials = 96 + 1 # include an extra trial for post learning calculations
	states = []

	ctx_dict = {0: np.array([0,1]),
				1: np.array([2,3]),
				2: np.array([4,5]),
				3: np.array([6,7])}
	all_actions = np.arange(0,8)

	# let's do this thing
	for s in np.arange(n_states):

		# check if random seed provided for sim
		if rnd_seeds is not None:
			random.seed(rnd_seeds[s])
			np.random.seed(rnd_seeds[s])

		# generate new learning simulation
		state = generate_state(n_trials)
		# save params copy
		state.params = params 

		# begin simulation
		for t in range(n_trials - 1):
			
			state.idx = t

			# calculate rho, beta_g, beta_n
			state = calc_rho(state)	

			# pick an action and generate PE
			this_ctx = state.ctx_order[t]
			state.policy_subset(ctx_dict[this_ctx],all_actions,this_ctx) 

			# update critic with PE
			state.critic(alpha_c)	

			# update actors with PE 
			state.act(alpha_a,norm=norm,mag=mag,hebb=hebb) 	

			# full information if applicable
			if state.full_info[state.C[t]]:
				C_alt = np.delete(ctx_dict[this_ctx],state.C_in_ctx[t])[0]
				state.counterfactual_update(C_alt,alpha_c,alpha_a, T, norm, mag)

			# decay actors to prior?
			if decay_to_prior:
				state.decay_to_prior(gamma=decay_to_prior_gamma)

		# post learning!
		t = 95
		options = np.arange(0,8)
		combos = list(itertools.combinations(options, 2))*4 # all pair-wise combos, 4 times
		# np.random.shuffle(combos)

		# test performance
		noptions = 8
		choice_count = np.zeros(noptions)
		total_occurences = np.ones(noptions)*(noptions-1)*4
		for combo in combos:
			state.idx = t

			# calculate rho, beta_g, beta_n
			this_r = 0	
			state.rho[t] = this_r		
			state.beta_g[t] = np.max([0,beta*(1 + this_r)])
			state.beta_n[t] = np.max([0,beta*(1 - this_r)])

			# pick an action and generate PE
			this_ctx = state.ctx_order[t]
			state.policy_subset(np.array(combo),all_actions,this_ctx,post = True) 

			# store 
			choice = state.C[t]
			choice_count[choice] = choice_count[choice] + 1
		
		state.choice_count = choice_count
		state.choice_rate = choice_count/total_occurences

		# save state learning
		states.append(state)

	return states
