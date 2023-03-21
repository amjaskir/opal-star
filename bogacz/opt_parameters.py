##################################################################
# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
#
# Decription: Finds the parameters that have the largest average
# AUC of learning and reward across rich and lean environments
# for OpAL* without DA modulation for 2-option environment
#
# Name: bogacz/opt_parameters.py
# Use: python3 opt_parameters.py $SLURM_ARRAY_TASK_ID
##################################################################

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import scipy.stats as stats
from sklearn import metrics

# get my helper modules
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../helpers'))
import params

def getAvg(learn_curve,reward_curve,n_trials):
	[avg_sm,sem_sm] = learn_curve
	[avg_r,sem_r] = reward_curve

	# calc AUC of both
	x = np.arange(n_trials)
	avg_learn = np.mean(avg_sm[0:n_trials])
	avg_reward = np.mean(avg_r[0:n_trials])

	return avg_learn, avg_reward

def main(pltn):
	"""
	pltn		number of trials to plot
	"""

	#######################################################
	# Various settings - restriction, full, anneal
	# restrict alpha actor to range with behaving curves
	root_me = "complexity"
	n_trials = 500
	n_states = 5000
	par_key = "ALU2"
	#######################################################

	param_comb = params.get_params_all(par_key)
	
	n_levels = 1 # number of levels of complexity
	full = True
	if full:
		add_on = "_FULL"
	else:
		add_on = ""

	#create directory for complex graphs when non-existing
	save_path = "results/%s_%d_%d%s/opt_res/ntrials%d/"\
		%(root_me,n_trials,n_states,add_on,pltn)
	os.makedirs(save_path, exist_ok=True)

	path_rich = "results/%s_%d_%d%s/rich/rich/k_0.0/mod_constant/" \
		%(root_me,n_trials,n_states,add_on)
	path_lean = "results/%s_%d_%d%s/lean/lean/k_0.0/mod_constant/" \
		%(root_me,n_trials,n_states,add_on)

	#intialize parameters
	maxAvg_rich = 0
	maxAlpha_rich = 0
	maxBeta_rich = 0
	maxEpsilon_rich = 0
	maxLambda_rich = 0

	maxAvg_lean = 0
	maxAlpha_lean = 0
	maxBeta_lean = 0
	maxEpsilon_lean = 0
	maxLambda_lean = 0

	for par in param_comb:

		r_par = np.round(par,5)
		alpha, epsilon, lbda, beta = r_par

		if os.path.exists(path_rich + "params_" + str(r_par) + ".pkle"):

			# get rich unmodulated data 
			_, _, learn_curveR, reward_curveR = \
			pickle.load(open(path_rich + "params_" + str(r_par) + ".pkle","rb"))
			avg_learnR, avg_rewardR = getAvg(learn_curveR,reward_curveR,pltn)

			# get lean unmodulated data 
			_, _, learn_curveL, reward_curveL = \
			pickle.load(open(path_lean + "params_" + str(r_par) + ".pkle","rb"))
			avg_learnL, avg_rewardL = getAvg(learn_curveL,reward_curveL,pltn)

			# check lean and rich
			if avg_learnL > maxAvg_lean:
				maxAvg_lean = avg_learnL
				maxAlpha_lean = alpha
				maxBeta_lean = beta
				maxEpsilon_lean = epsilon
				maxLambda_lean = lbda

			if avg_learnR > maxAvg_rich:
				maxAvg_rich = avg_learnR
				maxAlpha_rich = alpha
				maxBeta_rich = beta
				maxEpsilon_rich = epsilon
				maxLambda_rich = lbda

		else:
			tried = str(path_base + "params_" + str(r_par) + ".pkle")
			print(tried)
			print("missing params", str(r_par))

	res = {}
	res["lean_max"] = maxAvg_lean
	res["lean_alpha"] = maxAlpha_lean
	res["lean_beta"] = maxBeta_lean
	res["lean_lambda"] = maxLambda_lean
	res["lean_epislon"] = maxEpsilon_lean
	res["rich_max"] = maxAvg_rich
	res["rich_alpha"] = maxAlpha_rich
	res["rich_beta"] = maxBeta_rich
	res["rich_lambda"] = maxLambda_rich
	res["rich_epislon"] = maxEpsilon_rich

	# save results
	pickle.dump(res,open(save_path + "res.pkle","wb"))

if __name__ == '__main__':
	main(int(sys.argv[1]))





