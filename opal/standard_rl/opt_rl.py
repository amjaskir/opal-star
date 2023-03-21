# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank

import numpy as np
from scipy.optimize import differential_evolution
import scipy.stats as stats
from sklearn import metrics

# dependency things
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

# my scripts
import rl
from rl import RL
import learn

def avg_sm(states):
	# calc learning curve according to softmax 
    # calc reward curve 
    # first option is always best option
    first = True
    for state in states:
        if first:
            sms = state.SM[:,0]
            first = False
        else:
            sms = np.vstack([state.SM[:,0],sms])
    avg_sm = np.mean(sms, axis = 0)
    sem_sm = stats.sem(sms, axis = 0)

    return (avg_sm,sem_sm)

def avg_sm_statefirst(states):
    sms = []
    for state in states:
        sms = np.append(sms,np.mean(state.SM[:,0]))

    avg_sm = np.mean(sms, axis = 0)
    sem_sm = stats.sem(sms, axis = 0)

    return (avg_sm,sem_sm)

def wrapper(params,args):
	"""
	Wrapper function to call simulation and return average 
	probability of selecting the best option
	"""
	n_states = 1000
	v0 = 0.5
	rmag = 1
	lmag = 0

	n_trials,n_options,UCB,samplemean = args
	# extract what I want
	if UCB and samplemean:
		gamma = params[0]
		new_p = [0.01,5.0] # not used for decision making
	elif UCB:
		gamma = params[1]
		alpha = params[0]
		new_p = [alpha,5] # beta not used for DM
	else:
		new_p = params

	# get envs
	rich_env = "80_10_%s" %(n_options)
	lean_env = "30_10_%s" %(n_options)

	# run for each environment
	states_rich = learn.simulate(new_p,n_states,n_trials,\
		v0=v0,env=rich_env,r_mag=rmag,l_mag=lmag,\
			UCB=UCB,UCB_samplemean=samplemean,gamma=gamma)
	states_lean = learn.simulate(new_p,n_states,n_trials,\
		v0=v0,env=lean_env,r_mag=rmag,l_mag=lmag,\
		UCB=UCB,UCB_samplemean=samplemean,gamma=gamma)

	# average results
	avg_rich, sem_rich = avg_sm(states_rich)
	avg_lean, sem_lean = avg_sm(states_lean)

	# passing to a minimizer
	return -np.mean([avg_rich,avg_lean])

def main(n_trials,n_options,UCB,samplemean):
	
	if UCB and samplemean:
		bounds = [(0,50)] # gamma
	elif UCB:
		bounds = [(0,1), (1, 50)] 	# alpha, gamma
	else:
		bounds = [(0,1), (1, 50)]	# alpha, beta

	args = [n_trials,n_options,UCB,samplemean]

	res = differential_evolution(wrapper, bounds, args=[args],\
		popsize=30,maxiter=2000)
	return res

if __name__ == '__main__':
	n_trials = int(sys.argv[1])
	n_options = int(sys.argv[2])
	UCB = bool(int(sys.argv[3]))
	samplemean = bool(int(sys.argv[4]))
	main(n_trials,n_options,UCB,samplemean)

