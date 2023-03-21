# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
import numpy as np
import scipy.stats as stats
import random

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# my modules
from opal import OpAL

def simulate(params,n_states,v0=0.0, mod = "zero", k=1., rho=0.0, d=1., sc = .2, rnd_seed = None):
	"""
	Simulates decision making in gambles of explicitly give values.
	Gamble reward values encoded relate to default, i.e. G/N
	weights explicityly set and no learning in this model.

	Based off paradigm from Rutledge et al 2015. 300 trials, 3 trial
	types (mixed,gain,loss) of 100 trials each.
	
	Can modulate DA at choice (rho) online by critic value or set to 
	specified constant value. Can also simulate L-Dopa drug effects
	by selectively amplifying high DA levels. 

	Inputs:
	params 	 - tuple of (alphaC, alphaA, beta)
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
	k - multiplier for rho modulation
	rho - if mod == "constant", desired rho level
	d = level of drug
	rnd_seed - sets random seed generator to desired seed at beginning of 
			   simulations

	Outputs:
	states - contains tracker for each simulated learning state
		   - see classes.py for more detail
	"""

	#############################################################################
	def calc_rho(state):
		"""
		calculates rho for the current trial and returns
		the updated state tracker for rho and respective betas
		"""
		if mod == "constant": 		# artificially set rho to specific value, e.g. rho = 0
			state.rho[t] = rho
		elif mod == "value":		# EV of current gamble
			state.rho[t] = state.offer_V[t]*k
		elif mod == "value_drug":   # modulate rho by value of the gamble + drug
			if state.offer_V[t] > 0:
				state.rho[t] = state.offer_V[t]*k*(1. + thisd)
			else:
				state.rho[t] = state.offer_V[t]*k
		elif mod == "PE":
			state.rho[t] = state.PE_tr[t]*k
		elif mod == "PE_drug":
			if state.PE_tr[t] > 0:
				state.rho[t] = state.PE_tr[t]*k*(1. + thisd)
			else:
				state.rho[t] = state.PE_tr[t]*k
		else:
			err = 'Invalid value given for arg mod. \"%s\" given' %mod
			raise Exception(err)

		state.beta_g[t] = np.max([0,beta*(1 + state.rho[t])])
		state.beta_n[t] = np.max([0,beta*(1 - state.rho[t])])
		return state

	def get_gamble(state):
		""" 
		Sets G/N and V explicitly according to trial type
		"""
		ttype = state.trial_type[t]

		# gamble_mags[0] = successful gamble
		if ttype == 0:		# gain trial
			sure_mag = np.random.uniform(.3,.55)
			multiplier = stats.truncnorm.rvs(1.6,4)
			state.r_mag = np.min([sure_mag*multiplier,2.75]) + np.zeros(1) #make array
			state.l_mag = 0. + np.zeros(1)

			# explicitly set G/N values relative to sure thing
			# gamble, cost - I lose sure thing
			state.QG[t] = state.r_mag - sure_mag
			state.QN[t] = sure_mag

		elif ttype == 1:	# loss trial
			sure_mag = -1.*np.random.uniform(.3,.55)
			multiplier = stats.truncnorm.rvs(1.6,4)
			state.r_mag = 0. + np.zeros(1)
			state.l_mag = np.max([sure_mag*stats.truncnorm.rvs(1.6,4),-2.75]) + np.zeros(1)

			# explicitly set G/N values
			# gamble, benefits - I don't lose money by avoiding sure thing
			state.QG[t] = -1.*sure_mag
			state.QN[t] = -1.*(state.l_mag - sure_mag)

		else:				# mixed trial
			sure_mag = 0.
			multiplier = stats.truncnorm.rvs(.5,5)
			state.r_mag = np.random.uniform(.4,.75) + np.zeros(1)
			state.l_mag = np.max([-1*state.r_mag*multiplier, -2.75]) + np.zeros(1)

			# explicitly set G/N values
			# gamble, benefits and costs encoded relative to sure thing
			state.QG[t] = state.r_mag
			state.QN[t] = -1*state.l_mag

		# track trial information
		state.sure_mags[t] = sure_mag
		state.r_mags[t] = state.r_mag
		state.l_mags[t] = state.l_mag 
		state.multiplier[t] = multiplier
		state.advantage[t] = np.mean([state.r_mag,state.l_mag]) - sure_mag

		# offer value is the average of certain and EV of gamble
		state.offer_V[t] = np.mean([sure_mag,np.mean([state.r_mag,state.l_mag])])
		#pgamble = state.SM[t-1]
		#state.V[t] = (1-pgamble)*sure_mag + pgamble*np.mean([state.r_mag,state.l_mag])
		#state.V[t] = np.mean([state.r_mag,state.l_mag]) - sure_mag # this is advantage of gamble
		
		# generate trial level PE and update
		# this is tracking the average value of offers
		# critic tracks average value of SELECTING the gamble
		state.PE_tr[t] =  state.offer_V[t] - state.V_tr[t]
		state.V_tr[t+1] = state.V_tr[t] + alpha_c*state.PE_tr[t]

		return state


	# generates nStates states according to the specified environment
	# returns generated states and the average EV of each state type
	def generate_state():
		n_options = 1
		probs = .5
		r_mag = l_mag = 0	# we will set these later
		
		n_per_type = 100
		# 0 - gain, 1 - loss, or 2 - mixed trials
		trial_type = np.concatenate((np.zeros(n_per_type),\
			np.zeros(n_per_type)+1., \
			np.zeros(n_per_type)+2.))
		random.shuffle(trial_type)
		new_state = OpAL(n_trials, "S", v0, n_options, probs, \
					r_mag, l_mag, trial_type = trial_type)
		return new_state
	#############################################################################

	# define parameters
	alpha_c, alpha_a, beta = params
	n_trials = 300 # 100 trials for each trial type
	states = []

	# check if random seed provided
	if rnd_seed is not None:
		random.seed(rnd_seed)
		np.random.seed(rnd_seed)

	# let's do this thing, n_states is num subjects
	for s in np.arange(n_states):

		# generate new learning state
		state = generate_state()
		state.params = params 
		state.env = "Rutledge"

		thisd = np.random.normal(loc=d, scale=sc)  # individual drug effect
		while (thisd <= 0) or (thisd >= 1): 
			thisd = np.random.normal(loc=d, scale=sc)
		state.d = thisd
		for t in range(n_trials):
			state.idx = t
			state = get_gamble(state)	# set G/N and V according to trial type
			state = calc_rho(state)	    # calculate rho, beta_g, beta_n
			state.policy_gamble()		# pick an action and generate PE
			state.critic(alpha_c)		# update critic with PE if gamble chosen
			# NO ACTOR UPDATING - QG AND QN SET EXPLICITLY

		# save state learning
		states.append(state)

	return states