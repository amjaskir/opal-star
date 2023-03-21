##################################################################
# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
#
# Decription: Compare effects of DA modulation in ALU model
# Name: bogacz/hist_grid.py
##################################################################

import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import itertools
import scipy.stats as stats
from sklearn import metrics

# get my helper scripts
sys.path.insert(1, os.path.join(sys.path[0], '../helpers'))
import params
import aucplotter

def getAUC(learn_curve,reward_curve,n_trials):
	[avg_sm,sem_sm] = learn_curve
	[avg_r,sem_r] = reward_curve

	# calc AUC of both
	x = np.arange(n_trials)
	auc_learn = metrics.auc(x,avg_sm[0:n_trials])
	auc_reward = metrics.auc(x,avg_r[0:n_trials])

	return auc_learn, auc_reward

def main(env,n_trials,n_states,pltn):
	"""
	Graphs data outputed by grid_search.py into histograms

	env  		environment specified by environments.py
	k 			multiplier for modulation
	reprocess 	whether to reprocess cached graph data
	ext 		version to analyze
				"thesh10" 
				"origin" - original optimized parameters 
	"""

	t = time.time()
	# get params to plot
	par_key = "ALU2"
	param_comb = params.get_params_all(par_key) # need to convert into all_combos?
	# ks = params.get_ks(par_key)[1:] # don't include leading 0.
	ks = np.array([1.])
	alpha_as = np.unique(np.array(param_comb)[:,0]) # list of all unique alpha_as
	epsilons = np.unique(np.array(param_comb)[:,1]) # list of all unique epsilons
	lmbdas = np.unique(np.array(param_comb)[:,2]) 	# list of all unique lmbds
	betas = np.unique(np.array(param_comb)[:,3]) 	# list of all unique betas


	for k in ks:
		path_base = "./results/%s_%d_%d/k_0.0/mod_constant/" \
					%(env,n_trials,n_states)
		path = "./results/%s_%d_%d/k_%s/" \
					%(env,n_trials,n_states,k)
		save_pltn = path + "ntrials" + str(pltn) + "/"

		auc_learn_modT = []
		auc_learn_modF = []
		auc_reward_modT = []
		auc_reward_modF = []
		auc_diff_learn = []
		auc_diff_reward = []

		# save parameters to color diff by AUC graphs
		color_alpha = []
		color_epsilon = []
		color_lbda = []
		color_beta = [] 
		color_l_less_a = [] 

		titles = ["Alpha","Epsilon","Lambda",\
		"Beta","Lambda < Alpha"]
		saveas = ["diff_by_alpha","diff_by_epsilon",\
		"diff_by_lambda","diff_by_beta",\
		"diff_by_l_less_a"]
		# use coolwarm for boolean
		maps = ["plasma","plasma","plasma","plasma","coolwarm"]	

		for par in param_comb:
			r_par = np.round(par,5)

			# handle parameter ish
			alpha_a, epsilon, lbda, beta = r_par
			color_alpha.append(alpha_a)
			color_epsilon.append(epsilon)
			color_lbda.append(lbda)
			color_beta.append(beta)
			color_l_less_a.append(int(lbda < alpha_a))

			if os.path.exists(path + "mod_value/params_" + str(r_par) + ".pkle"):
				# get modulated data
				_, _, learn_curveT, reward_curveT = \
				pickle.load(open(path + "mod_value/params_" + str(r_par) + ".pkle","rb"))
				auc_learnT, auc_rewardT = getAUC(learn_curveT,reward_curveT,pltn)

				# get unmodulated data 
				_, _, learn_curveF, reward_curveF = \
				pickle.load(open(path_base + "params_" + str(r_par) + ".pkle","rb"))
				auc_learnF, auc_rewardF = getAUC(learn_curveF,reward_curveF,pltn)

				# save
				auc_learn_modT.append(auc_learnT)
				auc_learn_modF.append(auc_learnF)
				auc_reward_modT.append(auc_rewardT)
				auc_reward_modF.append(auc_rewardF)
				auc_diff_learn.append(auc_learnT - auc_learnF)
				auc_diff_reward.append(auc_rewardT - auc_rewardF)

				# save learning curves for parameter
				aucplotter.learn_curve(learn_curveT,auc_learnT,learn_curveF,auc_learnF,r_par,save_pltn,\
					w_auc=True,pltn=pltn)

				# TODO: FIGURE OUT IF I WANT TO PLOT LEARNING CURVES IN LARGER GRID

			else:
				tried = str(path + "mod_beta/params_" + str(r_par) + ".pkle")
				print(tried)
				print("missing params", str(r_par))

		colors = (color_alpha,color_epsilon,color_lbda,color_beta,color_l_less_a)


		# Graph everything
		what = "mod"
		aucplotter.graph_learn(auc_learn_modT,auc_learn_modF,save_pltn,what)
		aucplotter.graph_reward(auc_reward_modT,auc_reward_modF,save_pltn,what)
		aucplotter.graph_diff(auc_diff_learn,save_pltn,"Learning",what)
		aucplotter.graph_diff(auc_diff_reward,save_pltn,"Reward",what)
		aucplotter.graph_diff_by_AUC(np.array(auc_learn_modF),np.array(auc_diff_learn),\
			colors,titles,saveas,maps,\
			save_pltn,"Learning",what)
		aucplotter.graph_diff_by_AUC(np.array(auc_reward_modF),np.array(auc_diff_reward),\
			colors,titles,saveas,maps,\
			save_pltn,"Reward",what)

		elapsed = time.time() - t
		print("k %d done, elapsed: %f" %(k,elapsed))

if __name__ == '__main__':
	main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))





