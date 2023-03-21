##################################################################
# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
#
# Decription: Compare effects of DA modulation in bayesian opal model
#
# Name: opal/hist_grid_bayes.py
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

def getAUC(learn_curve,reward_curve,n_trials):
	[avg_sm,sem_sm] = learn_curve
	[avg_r,sem_r] = reward_curve

	# calc AUC of both
	x = np.arange(n_trials)
	auc_learn = metrics.auc(x,avg_sm[0:n_trials])
	auc_reward = metrics.auc(x,avg_r[0:n_trials])

	return auc_learn, auc_reward


def main(env_base,n_trials,n_states,pltn):
	"""
	Graphs data outputed by grid_search.py into histograms

	env_base  	environment specified by environments.py
	n_trials, n_states	specifies simulation
	pltn		number of trials to plot
	variant		saving folder name
	"""

	# get parameters
	par_key = "extrange"
	param_comb = params.get_params_all(par_key)
	alphas = np.unique(np.array(param_comb)[:,1]) # list of all unique alpha_as
	betas = np.unique(np.array(param_comb)[:,2]) 	# list of all unique betas
	n_levels = 8 # number of levels of complexity


	# look at curves for each level of complexity
	for level in np.arange(n_levels):
		# get environment name
		if level == 0:
			n_opt = ""              # no number for base environment
		else:   
			n_opt = "_" + str(level + 2)  # offset by 2
		
		env = env_base + n_opt
		path = "results/complexity_%d_%d/%s/%s/" \
			%(n_trials,n_states,env_base,env)

		# save in separate folder for the specified pltn
		save_pltn = path + "ntrials" + str(pltn) + "/"
		auc_learn_all = []
		auc_reward_all = []

		# save parameters to color diff by AUC graphs
		color_beta = []
		color_alphaa = []
		titles = ["Beta","Alpha A"]
		saveas = ["diff_by_auc_beta","diff_by_auc_alphaa"]
		maps = ["plasma","plasma"]	# use coolwarm for boolean

		plt.rcParams.update({'font.size': 8})
		fig_curv, ax_curv = plt.subplots(len(alphas),len(betas))
		fig_curv.set_size_inches(22, 17)
		fig_curv.tight_layout(pad=4.0, w_pad=4.0, h_pad=4.0)
		for par in param_comb:
			par2 = par[1:3]

			# handle parameter ish
			alpha_a, beta = par2
			color_alphaa.append(alpha_a)
			color_beta.append(beta)

			if os.path.exists(path + "params_" + str(par2) + ".pkle"):
				
				# get data
				_, _, learn_curve, reward_curve, rnd_seed = \
				pickle.load(open(path + "params_" + str(par2) + ".pkle","rb"))
				auc_learn, auc_reward = getAUC(learn_curve,reward_curve,pltn)

				auc_learn_all.append(auc_learn)
				auc_reward_all.append(auc_reward)

				# save learning curves for parameter
				# NOTE: this is just plotting the same line
				aucplotter.learn_curve(learn_curve,auc_learn,learn_curve,auc_learn,par2,save_pltn,\
					w_auc=True,pltn=pltn)

				# save learning curves in larger grid figure
				which_a = np.where(alphas == alpha_a)[0][0]
				which_b = np.where(betas == beta)[0][0]
				this_ax = ax_curv[which_a,which_b]
				aucplotter.learn_curve(learn_curve,auc_learn,learn_curve,auc_learn,par2,save_pltn,\
					w_auc=True,pltn=pltn,ax=this_ax)

			else:
				tried = str(path + "params_" + str(par2) + ".pkle")
				print(tried)
				print("missing params", str(par2))

		# graph AUC hist of all the data
		what = "mod"
		aucplotter.graph_learn(auc_learn_all,auc_learn_all,save_pltn,what)
		aucplotter.graph_reward(auc_reward_all,auc_reward_all,save_pltn,what)

		# save all curves in one plot
		plt.rcParams.update({'font.size': 22})
		plt.savefig(save_pltn + "learning", dpi = 400)
		plt.close()


if __name__ == '__main__':
	main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))





