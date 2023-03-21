##################################################################
# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
#
# Decription: Compare unmodulated models in rich and lean performance
# Name: opal/compare_env.py
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


def main(version,reprocess):
	"""
	Graphs data outputed by grid_search.py into histograms

	env  		environment specified by environments.py
	reprocess 	whether to reprocess cached graph data
	version 	ver of ALU ran	
				"thresh10" 
				"thresh20"
				"thresh10_v05", G/N init is .5
	"""

	if version == "std":
		path_lean = "results/grid_thresh10/lean/norm_False/k_0.0/mod_constant/" 
		path_rich = "results/grid_thresh10/rich/norm_False/k_0.0/mod_constant/" 
		# save norm comparisons in version folder
		path = "results/grid_thresh10/nF/" 
	elif version == "bayes":
		path_lean = "results/grid_bayes/lean/norm_True/k_0.0/l_100.0/mod_constant/" 
		path_rich = "results/grid_bayes/rich/norm_True/k_0.0/l_100.0/mod_constant/" 
		# save norm comparisons in version folder
		path = "results/grid_bayes/nT/" 
	else:
		err = 'Invalid value given for arg version. \"%s\" given' %version
		raise Exception(err)

	# check if cached data exists
	dump_it = path + "saved.pkle"
	if os.path.exists(dump_it) and not reprocess:
		auc_learn_rich,auc_learn_lean,auc_reward_rich,auc_reward_lean,\
		auc_diff_learn,auc_diff_reward = \
		pickle.load(open(dump_it,"rb"))

	else:
		# all param combinations
		if version == "bayes":
			param_comb = params.get_grid_bayes()
		else:
			param_comb = params.get_grid()

		auc_learn_rich = []
		auc_reward_rich = []
		auc_learn_lean = []
		auc_reward_lean = []
		auc_diff_learn = []
		auc_diff_reward = []

		for par in param_comb:

			if version == "bayes":
				par = par[1:3]

			if os.path.exists(path_rich + "params_" + str(par) + ".pkle"):
				# get normalized data
				auc_learnR, auc_rewardR, learn_curve, reward_curve = \
				pickle.load(open(path_rich + "params_" + str(par) + ".pkle","rb"))
				# get unnormalized data 
				auc_learnL, auc_rewardL, learn_curve, reward_curve = \
				pickle.load(open(path_lean + "params_" + str(par) + ".pkle","rb"))

				# save
				auc_learn_rich.append(auc_learnR)
				auc_learn_lean.append(auc_learnL)
				auc_reward_rich.append(auc_rewardR)
				auc_reward_lean.append(auc_rewardL)
				auc_diff_learn.append(auc_learnR - auc_learnL)
				auc_diff_reward.append(auc_rewardR - auc_rewardL)

			else:
				print(path + "mod_value/params_" + str(par) + ".pkle")
				print("missing params" + str(par))

		# save to regreaph later
		pickle.dump((auc_learn_rich,auc_learn_lean,auc_reward_rich,auc_reward_lean,\
			auc_diff_learn,auc_diff_reward), \
			open(dump_it, "wb"))

	# Graph everything
	what = "env"
	aucplotter.graph_learn(auc_learn_rich,auc_learn_lean,path,what)
	aucplotter.graph_reward(auc_reward_rich,auc_reward_lean,path,what)
	aucplotter.graph_diff(auc_diff_learn,path,"Learning",what)
	aucplotter.graph_diff(auc_diff_reward,path,"Reward",what)
	aucplotter.graph_diff_by_AUC(np.array(auc_learn_lean),np.array(auc_diff_learn),\
		[],[],[],[],path,"Learning",what)
	aucplotter.graph_diff_by_AUC(np.array(auc_reward_lean),np.array(auc_diff_reward),\
		[],[],[],[],path,"Reward",what)

if __name__ == '__main__':
	main(sys.argv[1],bool(int(sys.argv[2])))





