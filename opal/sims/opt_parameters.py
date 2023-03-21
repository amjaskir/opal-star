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
# Name: opal/opt_parameters.py
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
sys.path.insert(1, os.path.join(sys.path[0], '../../helpers'))
import params
import aucplotter

def getAvg(learn_curve,reward_curve,n_trials):
	[avg_sm,sem_sm] = learn_curve
	[avg_r,sem_r] = reward_curve

	# calc avg of both
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

	# anneal learning rate?
	anneal = True
	T = 100.0
	if anneal:
		base = "anneal_%d/" %T
		root_me = base + root_me

	n_trials = 500
	n_states = 5000

	#######################################################

	par_key = "extrange"
	param_comb = params.get_params_all(par_key)
	alpha_as = np.unique(np.array(param_comb)[:,1]) # list of all unique alpha_as
	betas = np.unique(np.array(param_comb)[:,2]) 	# list of all unique betas

	#create directory for complex graphs when non-existing
	save_path = "results/%s_%d_%d/opt_res/ntrials%d/"\
		%(root_me,n_trials,n_states,pltn)
	os.makedirs(save_path, exist_ok=True)

	path_rich = "results/%s_%d_%d/rich/rich/k_0.0/mod_constant/" \
		%(root_me,n_trials,n_states)
	path_lean = "results/%s_%d_%d/lean/lean/k_0.0/mod_constant/" \
		%(root_me,n_trials,n_states)

	#intialize parameters
	maxAvg_learn = 0
	maxAvg_reward = 0
	maxAlpha_learn = 0
	maxBeta_learn = 0
	maxAlpha_reward = 0
	maxBeta_reward = 0

	for par in param_comb:
		par2 = par[1:3]
		print("par: %s" %str(par2))

		# handle parameter ish
		alpha_a, beta = par2

		if os.path.exists(path_rich + "params_" + str(par2) + ".pkle"):

			# get rich unmodulated data 
			_, _, learn_curveR, reward_curveR, _ = \
			pickle.load(open(path_rich + "params_" + str(par2) + ".pkle","rb"))
			avg_learnR, avg_rewardR = getAvg(learn_curveR,reward_curveR,pltn)

			# get lean unmodulated data 
			_, _, learn_curveL, reward_curveL, _ = \
			pickle.load(open(path_lean + "params_" + str(par2) + ".pkle","rb"))
			avg_learnL, avg_rewardL = getAvg(learn_curveL,reward_curveL,pltn)

			# average lean and rich
			checkLearn = np.mean([avg_learnR,avg_learnL])
			if checkLearn > maxAvg_learn:
				maxAvg_learn = checkLearn
				maxAlpha_learn = alpha_a
				maxBeta_learn = beta

			checkReward = np.mean([avg_rewardR,avg_rewardL])
			if checkReward > maxAvg_reward:
				maxAvg_reward = checkReward
				maxAlpha_reward = alpha_a
				maxBeta_reward = beta

		else:
			tried = str(path_base + "params_" + str(par2) + ".pkle")
			print(tried)
			print("missing params", str(par2))

	res = {}
	res["learn_max"] = maxAvg_learn
	res["learn_alpha"] = maxAlpha_learn
	res["learn_beta"] = maxBeta_learn
	res["reward_max"] = maxAvg_reward
	res["reward_alpha"] = maxAlpha_reward
	res["reward_beta"] = maxBeta_reward

	# save results
	pickle.dump(res,open(save_path + "res.pkle","wb"))

if __name__ == '__main__':
	main(int(sys.argv[1]))





