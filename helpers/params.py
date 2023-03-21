##################################################################
# Helper script for grid searching and then retrieving parameter
# sweep for future analysis
#
# file: params.py
##################################################################

import numpy as np
import itertools

##################################################################
# OPAL PARAMETER GRIDS
##################################################################

def get_orig():
	# original parameters for opal in man mod scripts"""
	alpha_cs = np.round(np.array([.1]),1)
	alpha_as = np.round(np.array([.3]),1)
	betas = np.round(np.array([1.5]),1)
	all_combos = list(itertools.product(*[alpha_cs,alpha_as,betas]))

	return all_combos

def get_grid():
	""" return combination of all 
	parameters in grid search"""
	alpha_cs = np.round(np.arange(.1,1.1,.1),1)
	alpha_as = np.round(np.arange(.1,1.1,.1),1)
	betas = np.round(np.arange(.25,2.1,.25),2)
	all_combos = list(itertools.product(*[alpha_cs,alpha_as,betas]))

	return all_combos

def get_grid_bayes():
	""" return combination of all 
	parameters in grid search"""
	alpha_cs = np.array([.0])   # used as filler, bayesian critic doesn't use
	alpha_as = np.round(np.arange(.1,1.1,.1),1)
	betas = np.round(np.arange(.25,2.1,.25),2)
	lmbdas = np.arange(100,201,100)
	all_combos = list(itertools.product(*[alpha_cs,alpha_as,betas,lmbdas]))

	return all_combos

def get_grid_bayesC():
	""" return combination of all 
	parameters in grid search for varying complexity domains """
	alpha_cs = np.array([.0])   # used as filler, bayesian critic doesn't use
	alpha_as = np.round(np.arange(.1,1.1,.1),1)
	betas = np.round(np.arange(.25,2.1,.25),2)
	lmbdas = np.array([0,25,50,100,200,400])
	all_combos = list(itertools.product(*[alpha_cs,alpha_as,betas,lmbdas]))

	return all_combos

def get_grid_bayesC2():
	""" return combination of all 
	parameters in grid search for varying complex bayes"""
	alpha_cs = np.array([.0])   # used as filler, bayesian critic doesn't use
	alpha_as = np.round(np.arange(.1,1.1,.1),1)
	betas = np.round(np.arange(1.,20.1,1.),2)
	lmbdas = np.array([0.])
	all_combos = list(itertools.product(*[alpha_cs,alpha_as,betas,lmbdas]))

	return all_combos

def get_grid_bayesC3():
	""" return combination of all 
	parameters in grid search for varying complex bayes
	THIS IS THE MAIN ONE """
	alpha_cs = np.array([.0])   # used as filler, bayesian critic doesn't use
	alpha_as = np.round(np.arange(.1,1.1,.1),1)
	betas = np.round(np.arange(1.,20.1,2.),2)
	lmbdas = np.array([0.])
	all_combos = list(itertools.product(*[alpha_cs,alpha_as,betas,lmbdas]))

	return all_combos

def get_grid_bayesC4():
	""" return combination of all 
	parameters in grid search for varying complex bayes
	THIS IS THE MAIN ONE """
	alpha_cs = np.array([.0])   # used as filler, bayesian critic doesn't use
	alpha_as = np.round(np.arange(.05,1.01,.05),2)
	betas = np.round(np.arange(1.,30.1,2.),2)
	all_combos = list(itertools.product(*[alpha_cs,alpha_as,betas]))

	return all_combos

def get_grid_bayesSA():
	""" return combination of all 
	parameters in grid search for varying complex bayes
	ncombos = 380 """
	alpha_cs = np.array([.0])   # used as filler, bayesian critic doesn't use
	alpha_as = np.round(np.arange(.05,1.01,.05),2)
	betas = np.round(np.arange(1.,5.1,.5),2) 
	all_combos = list(itertools.product(*[alpha_cs,alpha_as,betas]))
	return all_combos

def get_grid_bayesSA_revision():
	""" return combination of all 
	parameters in grid search for varying complex bayes
	ncombos = 380 """
	alpha_cs = np.array([.0])   # used as filler, bayesian critic doesn't use
	alpha_as = np.round(np.arange(.05,1.01,.05),2)
	betas = np.round(np.arange(1.,10.1,.5),2) #expanded from manuscript simulations from 5-10
	all_combos = list(itertools.product(*[alpha_cs,alpha_as,betas]))
	return all_combos

def get_grid_rlSA():
	""" return combination of all 
	parameters in grid search for varying complex bayes
	ncombos = 1444 """
	alpha_cs = np.array([0.025,0.05,.1])
	alpha_as = np.round(np.arange(.05,1.01,.05),2)
	betas = np.round(np.arange(1.,10.1,.5),2)
	all_combos = list(itertools.product(*[alpha_cs,alpha_as,betas]))
	all_combos = [item for item in all_combos if item[0] <= item[1]] #only run if alpha c < alpha a
	return all_combos

def get_grid_rl():
	""" return combination of all 
	parameters in grid search for varying complex bayes
	ncombos = 520 """
	alphas = np.round(np.arange(.05,1.01,.05),2)
	betas = np.round(np.arange(2.,100.1,2.0),2) #np.round(np.arange(2.,100.1,2),2)
	all_combos = list(itertools.product(*[alphas,betas]))
	return all_combos

def get_beta_var():
	""" return combination of all 
	parameters in grid search for varying complex bayes """
	alpha_cs = np.array([.0])   # used as filler, bayesian critic doesn't use
	alpha_as = np.round(np.arange(.1,1.1,.1),1)
	betas = np.round(np.arange(1.,20.1,2.),2)
	lmbdas = np.array([25.,50.,100.,200.,400.])
	all_combos = list(itertools.product(*[alpha_cs,alpha_as,betas,lmbdas]))

	return all_combos

##################################################################
# ALU PARAMETER GRIDS 
##################################################################

def get_the_rest(param):
	""" get epsilon and lambda according 
	to cq,cs,alpha

	eq. pg 10, Moller and Bogacz 2019
	"finally, how can such parameters...be determined?"
	"""
	cq, cs, alpha, beta = param
	epsilon = (1 - cs*(1/cq - 1))/(1 + cs*(1/cq - 1))
	lbda = alpha*(1 - epsilon)/(2*cs)

	return (alpha,epsilon,lbda,beta)

def get_grid_ALU():
	""" return combination of all 
	parameters in grid search"""
	# cq = np.round(np.arange(.6,1.0,.1),1)  #approx 1, must be <1
	# cs = np.round(np.arange(.6,1.0,.1),1)  #nearly 1
	# alpha = np.round(np.arange(.1,1.1,.1),1)
	# beta = np.round(np.arange(5,21,5),1) 
	
	cq = np.round(np.arange(.7,1.01,.1),1)  	#approx 1, must be <1
	cs = np.round(np.arange(.7,1.01,.1),1)  #nearly 1
	alpha = np.round(np.arange(.05,0.5,.05),1)
	beta = np.round(np.arange(2.,100.1,2.0),2) 
	# alpha = np.round(np.arange(.05,1.01,.05),2) # expand
	# beta = np.round(np.arange(2.,100.1,2.0),2) 

	combos = list(itertools.product(*[cq,cs,alpha,beta]))

	# get all params using s
	all_combos = np.apply_along_axis(get_the_rest,1,combos)

	return all_combos

def get_grid_ALU_old():
	""" return combination of all 
	parameters in grid search"""
	cq = np.round(np.arange(.7,.9,.1),1)  #approx 1, must be <1
	cs = np.round(np.arange(.8,1.01,.1),1)  #nearly 1
	alpha = np.round(np.arange(.1,1.1,.1),1)
	beta = np.round(np.arange(1,20.1,2),1) 

	combos = list(itertools.product(*[cq,cs,alpha,beta]))

	# get all params using s
	all_combos = np.apply_along_axis(get_the_rest,1,combos)

	return all_combos

##################################################################
# General 
##################################################################

def get_ks(version):
	""" get the list of params for this job idx 
		divides combination list into split # of groups 
	"""

	# for all: k=0 means no modulation
	# get appropriate version
	if version == "bayes":
		ks = np.round(np.arange(1.,10.,1),1)
	# complexity calls below
	elif (version == "main") or (version == "betamod"):
		ks = np.array([0.])
		ks = np.append(ks,np.round(np.arange(1.,10.,2),1))
		ks = np.append(ks,[10.,20.,50.,100.])
	elif version == "flip":
		ks = np.array([0.])
		ks = np.append(ks,np.array([1.,5.,10.,20.,50.,100.,200.,500.]))
	elif (version == "extrange") or (version == "extrange_bmod") or (version == "extrange_flip") or (version == "bmod") or (version == "simplemod"):
		ks = np.array([0.])
		ks = np.append(ks,np.array([1.,5.,10.,20.,50.,100.,200.,500.]))
	elif (version == "lrate"):
		ks = np.array([0.])
		ks = np.append(ks,np.array([.1,.25,.5,.75,1,5,10]))
	elif version == "ALU":
		ks = np.round(np.arange(0,10.1),1) 
	elif version == "ALU2":
		ks = np.array([0.])
		ks = np.append(ks,np.array([1.,5.,10.,20.,50.,100.,200.,500.]))
	else:
		err = 'Invalid value given for arg version. \"%s\" given' %version
		raise Exception(err)

	return ks

def get_params(idx,split,version):
	""" get the list of params for this job idx 
		divides combination list into split # of groups 
	"""

	# get appropriate version
	if version == "bayes":
		all_combos = get_grid_bayes()
	# complexity calls below
	elif (version == "main") or (version == "betamod"):
		all_combos = get_grid_bayesC()
	elif version == "flip":
		all_combos = get_grid_bayesC2()
	elif (version == "extrange") or (version == "extrange_bmod") or (version == "extrange_flip") or (version == "lrate") or (version == "bmod"):
		all_combos = get_grid_bayesC3()
	elif (version == "simplemod"):
		all_combos = get_grid_bayesSA()
	elif (version == "bayesCrit"):
		all_combos= get_grid_bayesSA_revision()
	elif (version == "rlCrit"):
		all_combos = get_grid_rlSA()
	elif version == "beta_var":
		all_combos = get_beta_var()
	elif version == "std":
		all_combos = get_grid()
	elif (version == "ALU") or (version == "ALU2"):
		all_combos = get_grid_ALU()
	elif (version == "ALUold"):
		all_combos = get_grid_ALU_old()
	elif (version == "RL"):
		all_combos = get_grid_rl()
	else:
		err = 'Invalid value given for arg version. \"%s\" given' %version
		raise Exception(err)

	divide_n_conquer = np.array_split(all_combos, split)

	return divide_n_conquer[idx]

def get_params_all(version):
	""" get the list of params for this job idx 
		divides combination list into split # of groups 
	"""

	# get appropriate version
	if version == "bayes":
		all_combos = get_grid_bayes()
	# complexity calls below
	elif (version == "main") or (version == "betamod"):
		all_combos = get_grid_bayesC()
	elif version == "flip":
		all_combos = get_grid_bayesC2()
	elif (version == "extrange") or (version == "extrange_bmod") or (version == "extrange_flip") or (version == "lrate"):
		all_combos = get_grid_bayesC3()
	elif (version == "simplemod"):
		all_combos = get_grid_bayesSA()
	elif (version == "bayesCrit"):
		all_combos= get_grid_bayesSA_revision()
	elif (version == "rlCrit"):
		all_combos = get_grid_rlSA()
	elif version == "std":
		all_combos = get_grid()
	elif (version == "ALU") or (version == "ALU2"):
		all_combos = get_grid_ALU()
	elif (version == "ALUold"):
		all_combos = get_grid_ALU_old()
	elif (version == "RL"):
		all_combos = get_grid_rl()
	else:
		err = 'Invalid value given for arg version. \"%s\" given' %version
		raise Exception(err)

	return all_combos
