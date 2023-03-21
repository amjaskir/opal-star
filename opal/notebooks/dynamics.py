# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank

# Name: plot_dynamics.py
# Description: Aids to plot model dynamics of OpAL model overtime

import numpy as np
import random

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# my modules
from opal import OpAL
import plot_dynamics
import learning

def simulate(params,n_states,n_trials,p=0.5,v0=0.5,r_mag=1,l_mag=0, norm=True, mag=1,\
	hebb=True,crit = "SA",anneal=False,use_var=False, pgrad = False, T=100.0,rnd_seed = None):
	"""
	Simulates OpAL algorithm for a single option with specified probability of 
	reward and reward/loss magnitudes

	Inputs:
	params - tuple of (alphaC, alphaA, beta)
			 - alphaC, learning rate of the critic
			 - alphaA, learning rate of the actor
			 - beta, softmax inverse temp for policy
	n_states - number of states of learning to simulate
	n_trials - number of trials of learning within each state
	p - probability of reward
	r_mag - reward magnitude
	l_mag - loss magnitude
	decay - decay actor weights by specified prior and gamma
	prior - prior for actor values if decay is set to true
	gamma - strength of decay, gamma > 0 less decay < 0 more decay 

	Outpute:
	states - contains tracker for internal learning evolution for each
			 state simulated
		   - see classes.py for more detail
	"""
	
	# initialize parameters
	n_options = 1
	alpha_c, alpha_a, beta = params
	states = []

	# check if random seed provided
	if rnd_seed is not None:
		random.seed(rnd_seed)
		np.random.seed(rnd_seed)

	# let's do this thing
	for s in np.arange(n_states):
		# generate new learning state
		state = OpAL(n_trials,crit,v0,n_options,[p],[r_mag],[l_mag],anneal=anneal,use_var=use_var,pgrad=pgrad,T=T)
		for t in range(n_trials):	
			state.idx = t 			# update states internal time index
			state.policy_forced(0)	# take action/get feedback
			state.critic(alpha_c)	# update
			state.act(alpha_a,norm=norm,mag=mag,hebb=hebb)
		
		states.append(state)	#save state learning

	return states

def simulate_learning(params,n_states,n_trials,v0=0.0,crit="S",env = "80_10_2",\
	mod = "constant",k=8., rho=0.0, thresh = 0, rnd_seeds = None,\
	norm=False,norm_type=None,mag=1,hebb=True,variant = None,\
	anneal = False, pgrad = False, T = 100.0,\
        decay_to_prior = False, decay_to_prior_gamma = 1.,
	full=False,r_mag=1,l_mag=-1,
	use_var = False, phi = 3):
	
	"""
	Wrapper to call learning.py script
	"""

	states = learning.simulate(params,n_states,n_trials,v0=v0,crit=crit,env=env,\
		mod="constant",k=k,norm=norm,norm_type=norm_type,mag=mag,rnd_seeds=rnd_seeds,\
			anneal=anneal,pgrad=pgrad,T=T,use_var=use_var,\
				phi=phi,r_mag=r_mag,l_mag=l_mag,hebb=hebb,decay_to_prior=decay_to_prior,
				decay_to_prior_gamma=decay_to_prior_gamma)
	
	return states


def simulate_together(params,n_states,n_trials,p,v0=0.5,r_mag=1,l_mag=0, norm=True,mag=1,\
	hebb=True,crit = "SA",bound=False,lim=2,anneal=False,pgrad=False,T=100.0,rnd_seed = None):
	"""
	Simulates OpAL algorithm for a single option with specified probability of 
	reward and reward/loss magnitudes

	Inputs:
	params - tuple of (alphaC, alphaA, beta)
			 - alphaC, learning rate of the critic
			 - alphaA, learning rate of the actor
			 - beta, softmax inverse temp for policy
	n_states - number of states of learning to simulate
	n_trials - number of trials of learning within each state
	p - probability of rewards, array
	r_mag - reward magnitude, array
	l_mag - loss magnitude, array
	decay - decay actor weights by specified prior and gamma
	prior - prior for actor values if decay is set to true
	gamma - strength of decay, gamma > 0 less decay < 0 more decay 

	Outpute:
	states - contains tracker for internal learning evolution for each
			 state simulated
		   - see classes.py for more detail
	"""
	
	# initialize parameters
	n_options = len(p)
	alpha_c, alpha_a, beta = params
	states = []

	# check if random seed provided
	if rnd_seed is not None:
		random.seed(rnd_seed)
		np.random.seed(rnd_seed)

	# let's do this thing
	for s in np.arange(n_states):
		# generate new learning state
		state = OpAL(n_trials,crit,v0,n_options,p,r_mag,l_mag,anneal=anneal,pgrad=pgrad,T=T)
		for t in range(n_trials):	
			state.idx = t 			# update states internal time index
			state.policy_forced(0)	# take action/get feedback
			state.critic(alpha_c)	# update
			state.act(alpha_a,norm=norm,mag=mag,\
				hebb=hebb,bound=bound,lim=lim) #assumes rmac/lmag same for all

			# force the model to select the remaining options in sequence
			state.update_other_actions(alpha_c,alpha_a,norm=norm,\
				mag=(r_mag[0] - l_mag[0]),
				hebb=hebb,bound=bound,lim=lim)
		
		states.append(state)	#save state learning

	return states

