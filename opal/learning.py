# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
import numpy as np
import opal
import random
from opal import OpAL
from environments import get_probs
from scipy.stats import beta as beta_rv

import time

def simulate(params,n_states,n_trials,\
	v0=0.5,r_mag=1,l_mag=-1,env = "80_10_2",\
	crit="SA",\
	mod = "constant", k=20., phi = 1,
	rho=0.0, baselinerho = 0.0, threshRho = 0, threshRand = 0, 
	rnd_seeds = None,\
	norm=False,mag=1,norm_type = None,hebb=True,variant = None,\
	anneal = False, T = 100.0,use_var=False,\
	gamble = False, p_gamble = .25,\
	decay_to_prior = False, decay_to_prior_gamma = 1.,\
	switch = None, env_switch = "30_10_2",\
	forced_actions = None, forced_rewards = None):
	"""
	Simulates learning in specified environment, each a multiarm
	bandit where success receives +1 and failure receives -1, for 
	specified number of trials. Can modulate DA at choice (rho) online
	by critic value or set to specified constant value 

	Inputs:
	params 	 - tuple of (alphaC, alphaA, beta)
			 - alphaC, learning rate of the critic
			 - alphaA, learning rate of the actor
			 - beta, softmax inverse temp for policy

	n_states - number of states of learning to simulate
	n_trials - number of trials of learning within each state

	v0 		- critic initialization value
	r_mag 	- reward mag
	l_mag 	- loss mag
	env 	- specified environment with reward probabilities of options
		  		see environments.py for options
				  
	crit 	- "S" or "SA"
		   	- "S"  critic tracks overall state value
		   	- "SA" critic tracks state-action pairs
			- "Bayes-SA" Bayesian critic

	mod - "beta", meta-critic using beta dist of env value
		- "constant", rho set to constant value throughout 
		- "avg_value", if critic tracks state-action values, modulates rho
					   by average over all values
	k 	- multiplier for rho modulation
	phi - param for # of std above/below .5 to modulate rho
	rho - if mod == "constant", desired rho level
	baselinerho - fixed rho around which to modulate
	threshRho 	- time point before which rho is constant and set to specified rho 

	threshRand - time before which policy is random

	rnd_seed - sets random seed generator to desired seed at beginning of 
			   simulations

	norm 	- T/F, normalize actor learning PE between [0,1] according to specified mag
	mag 	- scalar, unsigned magnitude of largest feedback for normalization

	hebb 	- T/F, use three-hactor hebbian term
	variant - str, specifies OpAL variant, see code for description
				"flip", "bmod","lrate"

	anneal 		- anneal actor learning rates?
	T 			- magnitude of annealing
	use_var 	- use variance of meta-critic to modulate annealing?

	FIG 10 SPECIFIC PARAMS
	gamble 		- gamble scenario?
	p_gamble	- probability of gamble payoff

	APPENDIX FIGS
	switch - introduce switch points?
	env_switch - environment post switch point

	FIG 8C SPECIFIC PARAMS
	forced_actions - forced agent to take specific action on each trial
	forced_rewards - forced reward for each trial

	Outputs:
	states - contains tracker for each simulated learning state
			 See opal.py for state details
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

				# am I sufficiently above/below 50%?
				thresh = phi*state.std[t]
				cond1 = (mean - thresh) > .5 # lower bound above .5
				cond2 = (mean + thresh) < .5 # upper bound below .5
				sufficient = cond1 or cond2

				# is sufficiently above/below .5 
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

		# variants for comparison
		if variant is None:
			# standard value modulation
			this_r = state.rho[t]
			state.beta_g[t] = np.max([0,beta*(1 + this_r)])
			state.beta_n[t] = np.max([0,beta*(1 - this_r)])
		elif variant == "flip":
			# flip the sign of modulation
			this_r = state.rho[t]
			this_r = -1.*this_r
			state.beta_g[t] = np.max([0,beta*(1 + this_r)])
			state.beta_n[t] = np.max([0,beta*(1 - this_r)])
		elif variant == "bmod":
			# rather than change bg and bg assymmetrically, 
			# changing rho only changes the overall choice temp.
			absrho = np.abs(state.rho[t])
			state.beta_g[t] = np.max([0,beta*(1 + absrho)])
			state.beta_n[t] = np.max([0,beta*(1 + absrho)])
		elif variant == "flipbmod":
			# rather than change bg and bg assymmetrically, 
			# changing rho only changes the overall choice temp.
			absrho = np.abs(state.rho[t])
			state.beta_g[t] = np.max([0,beta*(1 + absrho)])
			state.beta_n[t] = np.max([0,beta*(1 + absrho)])
		elif variant == "lrate":
			# rather than change bg and bg assymmetrically, 
			# changing rho only changes asymetry in learning rate
			state.beta_g[t] = beta  # only change alpha asym
			state.beta_n[t] = beta
		else:
			err = 'Invalid value given for arg variant. \"%s\" given' %variant
			raise Exception(err)

		return state

	def generate_state_gamble():
			"""
			intialize state for learning according to specified environment type

			r_mag = 1  # get double s.t.
			l_mag = -1 # lose s.t.
			"""
			ntrials = n_trials
			if env == "high":
				n_options = 1
				rmag = np.zeros(n_options) + 1
				lmag = np.zeros(n_options) + -1
				probs = np.random.uniform(.5,1.)
			elif env == "low":	
				n_options = 1	
				rmag = np.zeros(n_options) + 1	
				lmag = np.zeros(n_options) + -1					
				probs = np.random.uniform(0.,.5)
			elif env == "highMag":
				n_options = 1
				rmag = np.zeros(n_options) + np.random.uniform(1.,2.)
				lmag = np.zeros(n_options) + -1
				probs = 0.5
			elif env == "lowMag":	
				n_options = 1	
				rmag = np.zeros(n_options) + np.random.uniform(0.,1.)	
				lmag = np.zeros(n_options) + -1					
				probs = 0.5
			elif env == "zalocusky":
				n_options = 1
				rmag = np.zeros(n_options) + r_mag 
				lmag = np.zeros(n_options) + l_mag
				probs = p_gamble
			elif (env == "ECincreasing") or (env == "ECdecreasing"):
				n_options = 1
				probs = .5	

				# common trials
				common_X = np.array([7.13, 7.26, 7.37, 7.49, 7.62, 7.76, 7.87, 7.99])
				np.random.shuffle(common_X)
				n_common = len(common_X)
				common_C = np.zeros(n_common) + 2.7
				# encoded relative to certain outcome, C
				common_rmag = common_X - common_C
				common_lmag = -1*common_C

				# set uncommon trials
				ntrials = 300 
				n_uncommon = ntrials - n_common
				if env == "ECincreasing":
					# 0, P:1/50		X: [2,4.5)	C: [1,2.25)
					# 1, P:49/50	X: [4.5,8]	C: [2.25,4]
					# NOTE: ignored inclusive/exclusive 
					Xrand = np.random.binomial(1,49/50,n_uncommon)
					Crand = np.random.binomial(1,49/50,n_uncommon)
					def increasing_X(bi):
						if bi == 0:
							return np.random.uniform(2,4.5) 
						else:
							return np.random.uniform(4.5,8)
					def increasing_C(bi):
						if bi == 0:
							return np.random.uniform(1,2.25) 
						else:
							return np.random.uniform(2.25,4)
					uncommon_X = np.around([increasing_X(X) for X in Xrand],2)
					uncommon_C = np.around([increasing_C(C) for C in Crand],2)	
				else:
					# 0, P:49/50	X: [2,5.5]	C: [1,2.75]
					# 1, P:1/50		X: (5.5,8]	C: (2.75,4]
					# NOTE: ignored inclusive/exclusive 
					Xrand = np.random.binomial(1,1/50,n_uncommon)
					Crand = np.random.binomial(1,1/50,n_uncommon)
					def decreasing_X(bi):
						if bi == 0:
							return np.random.uniform(2,5.5) 
						else:
							return np.random.uniform(5.5,8)
					def decreasing_C(bi):
						if bi == 0:
							return np.random.uniform(1,2.75) 
						else:
							return np.random.uniform(2.75,4)
					uncommon_X = np.around([decreasing_X(X) for X in Xrand],2)
					uncommon_C = np.around([decreasing_C(C) for C in Crand],2)	
							
				rmag = uncommon_X - uncommon_C 
				lmag = -1*uncommon_C
				Xs = uncommon_X
				Cs = uncommon_C	

				# insert common in the uncommon trials
				insert_common = np.array([90,120,150,180,210,240,270,300])-1
				for arr_idx, idx in enumerate(insert_common):
					rmag = np.insert(rmag,idx,common_rmag[arr_idx])
					lmag = np.insert(lmag,idx,common_lmag[arr_idx])
					Xs = np.insert(Xs,idx,common_X[arr_idx])
					Cs = np.insert(Cs,idx,common_C[arr_idx])
			else:
				err = 'Invalid value given for arg env. %s given' %env
				raise Exception(err)

			new_state = OpAL(ntrials, crit, v0, n_options, probs, rmag, lmag,\
				anneal=anneal,T=T,norm_type=norm_type)

			if (env == "ECincreasing") or (env == "ECdecreasing"):
				new_state.QG = rmag
				new_state.QN = -1*lmag
				new_state.uncommon_X = uncommon_X
				new_state.uncommon_C = uncommon_C
				new_state.common_C = common_C
				new_state.common_X = common_X
				new_state.Xs = Xs
				new_state.Cs = Cs
			return new_state

	def generate_state():
		"""
		intialize state for learning according to specified probs
		assumes same r_mag and l_mag for each option
		"""

		if gamble:
			return generate_state_gamble()

		probs = get_probs(env)
		n_options = len(probs)

		# feedback for agent
		rmag = np.zeros(n_options) + r_mag
		lmag = np.zeros(n_options) + l_mag

		new_state = OpAL(n_trials,crit,v0,n_options,probs,rmag,lmag,\
							anneal=anneal,use_var=use_var,T=T,norm_type=norm_type)
		return new_state
	
	##########################################################################
	
	# define parameters
	alpha_c, alpha_a, beta = params
	states = []

	# let's do this thing
	for s in np.arange(n_states):

		# check if random seed provided for sim
		if rnd_seeds is not None:
			random.seed(rnd_seeds[s])
			np.random.seed(rnd_seeds[s])

		# generate new learning simulation
		state = generate_state()
		# save params copy
		state.params = params 
		state.env = env

		# begin simulation
		for t in range(n_trials):
			
			state.idx = t

			# switch point?
			if (switch != None) and (t == switch):
				state.probs = get_probs(env_switch)

			# calculate rho, beta_g, beta_n
			state = calc_rho(state)	

			# pick an action and generate PE
			if gamble:
				state.policy_gamble(thresh=threshRand)
			else:
				state.policy(thresh=threshRand,forced_actions=forced_actions,forced_rewards=forced_rewards,state_idx=s) 

			# update critic with PE
			state.critic(alpha_c,gamble=gamble)	

			# update actors with PE 
			# with efficient coding, actors set explicitly, so no need to update
			if (str(env) != "ECincreasing") and (str(env) != "ECdecreasing"):
				state.act(alpha_a,norm=norm,mag=mag,hebb=hebb,\
					var=variant,gamble=gamble) 	

			# decay actors to prior?
			if decay_to_prior:
				state.decay_to_prior(gamma=decay_to_prior_gamma)

		# save state learning
		states.append(state)

	return states
