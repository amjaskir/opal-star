##################################################################
# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
#
# Decription: Compare effects of DA modulation in bayesian opal model
#
# Name: opal/sims/stats_analysis.py
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

def getAUC(learn_curve,reward_curve,pltn):
	[avg_sm,sem_sm] = learn_curve
	[avg_r,sem_r] = reward_curve

	# calc AUC of both
	x = np.arange(pltn)
	auc_learn = metrics.auc(x,avg_sm[0:pltn])
	auc_reward = metrics.auc(x,avg_r[0:pltn])

	return auc_learn, auc_reward


def main(n_trials,n_states,pltn,variant=None):
	"""
	Graphs data outputed by grid_search.py into histograms

	env_base  	environment specified by environments.py
	n_trials, n_states	specifies simulation
	pltn		number of trials to plot
	variant		saving folder name
	"""
	print("starting script")
	sys.stdout.flush()


	#######################################################
	# This is copied from complexity.py
	par_key = "simplemod"
	param_comb = params.get_params_all(par_key)
	alpha_as = np.unique(np.array(param_comb)[:,1]) # list of all unique alpha_as
	betas = np.unique(np.array(param_comb)[:,2]) 	# list of all unique betas
	print(alpha_as)
	print(betas)
	ks = params.get_ks(par_key)[1:] 

	root_me = "simplemod/"
	n_levels = 5 #9 # number of levels of complexity
	mod = "mod_beta"

	# constant parameters for learning
	crit = "Bayes-SA"
	norm = True	# normalization
	root_me = root_me + crit + "/"

	# reward mag and loss mag
	# same for each option
	r_mag = 1
	l_mag = 0
	v0 = 0.5*r_mag + .5*l_mag
	mag = r_mag - l_mag
	base = "rmag_%d_lmag_%d_mag_%d/" %(r_mag,l_mag,mag)
	root_me = root_me + base

	# only modify sufficiently above 50% by phi*std
	# now coded that mod is only when sufficient
	phi = 1.0
	base = "phi_%.1f/" %(phi)
	root_me = root_me + base

	# use_std for K MODULATION 
	use_std = False
	base = "usestd_%r/" %(use_std)
	root_me = root_me + base

	# Use expected value (vs. beta mean) for mod
	exp_val = False
	base = "exp_val_%r/" %(exp_val)
	root_me = root_me + base

	# anneal learning rate?
	anneal = True
	T = 10.0
	base = "anneal_%r_T_%d/" %(anneal,T)
	root_me = root_me + base

	#######################################################
	# preset k
	k = 20.0

	# environments
	envs = ["80","30"]
	#######################################################

	#create directory for complex graphs when non-existing
	save_path = "results/%s/trials%d_sims%d/stats_summaries/k%d/"\
		%(root_me,n_trials,n_states,k)
	if variant is not None:
		save_path = save_path + variant + "/"
	print("Save path:" + save_path)
	os.makedirs(save_path, exist_ok=True)

	# initialize dictionaries
	all_stats = {}
	all_stats_wil = {}
	all_stats_wir = {}
	all_stats_wel = {}
	all_stats_wer = {}
	all_stats_tl = {}
	all_stats_tr = {}
	all_stats_t1r = {}
	all_stats_t1l = {}
	for env_base in envs:
		print("env: %s" %env_base)

		for level in np.arange(n_levels):
			print("level: %s" %level)

			# get environment name
			env = "%s_%d_%d" %(env_base,10,level + 2)
			if variant is None:
				path_bal = "./results/%strials%d_sims%d/%s/k_0.0/mod_constant/" \
				%(root_me,n_trials,n_states,env)
			else:
				# variants share root with modulation sims
				path_bal = "./results/%s/trials%d_sims%d/%s/k_%s/" \
				%(root_me,n_trials,n_states,env,k)
				append = "mod_beta/%s/" %(variant)
				path_bal = path_bal + append
			path_mod = "./results/%s/trials%d_sims%d/%s/k_%s/mod_beta/" \
				%(root_me,n_trials,n_states,env,k)

			# AUCs of learning curves and reward curves
			# for all parameters
			learn_mod_all = []		# DA modulated
			reward_mod_all = []
			learn_bal_all = []		# DA balanced
			reward_bal_all = []
			learn_diff = []
			reward_diff = []

			idx = 0
			for par in param_comb:
				par2 = par[1:3]
				# print("par: %s" %str(par2))
				# handle parameter ish
				alpha_a, beta = par2

				if os.path.exists(path_mod + "params_" + str(par2) + ".pkle"):

					if (idx == 0):
						p1 = path_mod + "params_" + str(par2) + ".pkle"
						p2 = path_bal + "params_" + str(par2) + ".pkle"
						print("opal path: %s" %p1)
						print("var path: %s" %p2)
						idx = 1
					# get modulated DA data
					_, _, learn_curve_mod, reward_curve_mod, rnd_seed = \
					pickle.load(open(path_mod + "params_" + str(par2) + ".pkle","rb"))
					auc_learn_mod, auc_reward_mod = getAUC(learn_curve_mod,reward_curve_mod,pltn)

					# get balanced DA data 
					_, _, learn_curve_bal, reward_curve_bal, rnd_seed = \
					pickle.load(open(path_bal + "params_" + str(par2) + ".pkle","rb"))
					auc_learn_bal, auc_reward_bal = getAUC(learn_curve_bal,reward_curve_bal,pltn)

					# save
					learn_mod_all.append(auc_learn_mod)
					learn_bal_all.append(auc_learn_bal)
					reward_mod_all.append(auc_reward_mod)
					reward_bal_all.append(auc_reward_bal)
					learn_diff.append(auc_learn_mod - auc_learn_bal)
					reward_diff.append(auc_reward_mod - auc_reward_bal)

				else:
					tried = str(path_mod + "params_" + str(par2) + ".pkle")
					print(tried)
					print("missing params", str(par2))

			# run the stats
			# DA mod is first input
			all_stats_wil[env]	= stats.wilcoxon(learn_mod_all,learn_bal_all)
			all_stats_wir[env]	= stats.wilcoxon(reward_mod_all,reward_bal_all)
			all_stats_tl[env]	= stats.ttest_rel(learn_mod_all,learn_bal_all)
			all_stats_tr[env]	= stats.ttest_rel(reward_mod_all,reward_bal_all)
			all_stats_wel[env]	= stats.ttest_ind(learn_mod_all,learn_bal_all,equal_var=True)
			all_stats_wer[env]	= stats.ttest_ind(reward_mod_all,reward_bal_all,equal_var=True)
			all_stats_t1l[env]	= stats.ttest_1samp(learn_diff,0)
			all_stats_t1r[env]	= stats.ttest_1samp(reward_diff,0)

			print("env: %s complete" %env)


		print("env: %s complete" %env_base)

	# combine into one dictionary for saving
	all_stats["wilcoxon_learn"] = all_stats_wil
	all_stats["wilcoxon_reward"] = all_stats_wir
	all_stats["ttest_learn"] = all_stats_tl
	all_stats["ttest_reward"] = all_stats_tr
	all_stats["welsch_learn"] = all_stats_wel
	all_stats["welsch_reward"] = all_stats_wer
	all_stats["ttest1_learn"] = all_stats_t1l
	all_stats["ttest1_reward"] = all_stats_t1r

	# save the data
	pickle.dump(all_stats,open(save_path + "ntrials%d_stats.pickle" %(pltn),"wb"))


if __name__ == '__main__':
	if len(sys.argv) > 4:
		main(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),sys.argv[4])
	else:
		main(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))





