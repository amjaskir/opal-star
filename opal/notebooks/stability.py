import numpy as np
import random

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# my modules
from opal import OpAL

def simulate_Bogacz2019(n_trials, alpha_a, alpha_c, V0, beta=1, \
	mag=1, norm=False, anneal=False, use_var=False, pgrad=False, T=100.0):
	
	# initialize parameters
	nOptions = 1
	probs = 1
	r_mag = 2
	l_mag = -1
	state = OpAL(n_trials,"SA",V0,nOptions,[probs],[r_mag],[l_mag],anneal=anneal,use_var=use_var,pgrad=pgrad,T=T)


	# run OpAL
	for t in range(n_trials):
		# alternate cost and loss trials
		if np.mod(t,2) == 0:
			state.probs = [0]
		else:
			state.probs = [1]

		# take action/get feedback
		state.policy_forced(0)

		# update
		state.critic(alpha_c)
		state.act(alpha_a,norm=norm,mag=mag)

		# update states internal time index
		state.idx = state.idx + 1

	return state


def simulate_appendix2(n_trials, alpha_a, alpha_c, V0, beta=1, \
	mag=1, norm=False, anneal=False, use_var = False,pgrad= False, decay_to_prior = False, decay_to_prior_gamma = 1., T=100.0,version="OpAL"):
	
	# initialize parameters
	nOptions = 1
	probs_init = 0.1		# starting probability
	r_mag = 1
	l_mag = 0
	probs = np.zeros(n_trials)
	if version == "OpAL*":
		state = OpAL(n_trials,"Bayes-SA",np.array([0.,0.]),nOptions,[probs_init],[r_mag],[l_mag],anneal=anneal,use_var=use_var,pgrad=pgrad,T=T)
	else:
		state = OpAL(n_trials,"SA",V0,nOptions,[probs_init],[r_mag],[l_mag],anneal=anneal,use_var=use_var,pgrad=pgrad,T=T)


	# run OpAL
	for t in range(n_trials):
		# random walk of reward probability
		state.probs[0] = state.probs[0] - 0.1*(state.probs[0]-0.5) + 0.1*np.random.randn()
		if state.probs[0] > 1:
			state.probs[0] = 1
		elif state.probs[0] < 0:
			state.probs[0] = 0
		probs[t] = state.probs[0]	# store to analyze later
 
		# take action/get feedback
		state.policy_forced(0)

		# update
		state.critic(alpha_c)
		state.act(alpha_a,norm=norm,mag=mag)

                # decay actors to prior?
		if decay_to_prior:
			state.decay_to_prior(gamma=decay_to_prior_gamma)

		# update states internal time index
		state.idx = state.idx + 1
	
	# store all the probs experienced
	state.probs = probs

	return state
