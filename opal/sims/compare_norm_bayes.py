##################################################################
# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
#
# Decription:Compare effect of normalized and unnormalized
# bayesian OpAL model (without dopamine modulation)
#
# Name: opal/compare_norm_bayes.py
##################################################################

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import scipy.stats as stats

# get my helper modules
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../../helpers'))
import params
import aucplotter


def main(env,reprocess):
	"""
	Graphs data outputed by grid_search.py into histograms

	env  		environment specified by environments.py
	norm  		whether PE is normalized or not
	k 			multiplier for modulation
	reprocess 	whether to reprocess cached graph data
	ext 		subset of parameters to plot
				"all" 
				"origin" - original optimized parameters 
	"""

	ext = "bayes"

	# note - this lambda is arb, either 100 or 200 will be fine, should check both
	path_normT = "results/grid_%s/%s/norm_%s/k_0.0/l_200.0/mod_constant/" \
				%(ext,env,"True")
	path_normF = "results/grid_%s/%s/norm_%s/k_0.0/l_200.0/mod_constant/" \
				%(ext,env,"False")
	# save norm comparisons in env folder
	path = "results/grid_%s/%s/" %(ext,env)

	# check if cached data exists
	dump_it = path + "saved.pkle"
	if os.path.exists(dump_it) and not reprocess:
		# TODO
		auc_learn_normT,auc_learn_normF,auc_reward_normT,auc_reward_normF,\
		auc_diff_learn,auc_diff_reward = \
		pickle.load(open(dump_it,"rb"))

	else:
		# all param combinations
		param_comb = params.get_grid_bayes()

		auc_learn_normT = []
		auc_reward_normT = []
		auc_learn_normF = []
		auc_reward_normF = []
		auc_diff_learn = []
		auc_diff_reward = []

		for par in param_comb:

			if os.path.exists(path_normT + "params_" + str(par[1:3]) + ".pkle"):
				# get normalized data
				auc_learnT, auc_rewardT, learn_curve, reward_curve = \
				pickle.load(open(path_normT + "params_" + str(par[1:3]) + ".pkle","rb"))
				# get unnormalized data 
				auc_learnF, auc_rewardF, learn_curve, reward_curve = \
				pickle.load(open(path_normF + "params_" + str(par[1:3]) + ".pkle","rb"))

				# save
				auc_learn_normT.append(auc_learnT)
				auc_learn_normF.append(auc_learnF)
				auc_reward_normT.append(auc_rewardT)
				auc_reward_normF.append(auc_rewardF)
				auc_diff_learn.append(auc_learnT - auc_learnF)
				auc_diff_reward.append(auc_rewardT - auc_rewardF)

			else:
				print("missing params" + str(par[1:3]))

		# save to regreaph later
		pickle.dump((auc_learn_normT,auc_learn_normF,auc_reward_normT,auc_reward_normF,\
			auc_diff_learn,auc_diff_reward), \
			open(dump_it, "wb"))

	# Graph everything
	what = "norm"
	aucplotter.graph_learn(auc_learn_normT,auc_learn_normF,path,what)
	aucplotter.graph_reward(auc_reward_normT,auc_reward_normF,path,what)
	aucplotter.graph_diff(auc_diff_learn,path,"Learning",what)
	aucplotter.graph_diff(auc_diff_reward,path,"Reward",what)
	aucplotter.graph_diff_by_AUC(np.array(auc_learn_normF),np.array(auc_diff_learn),\
		[],[],[],[],path,"Learning",what)
	aucplotter.graph_diff_by_AUC(np.array(auc_reward_normF),np.array(auc_diff_reward),\
		[],[],[],[],path,"Reward",what)

if __name__ == '__main__':
	main(sys.argv[1],bool(int(sys.argv[2])))





