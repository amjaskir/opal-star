##################################################################
# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
#
# Decription: Compares learning of two variations of
# opal bayes simulations, for example one with standard
# value modulation and one with modulation sign flipped
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

# get my helper modules
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../../helpers'))
import params
import aucplotter


def main(env_base,root_me1,root_me2):
	"""
	Graphs data outputed by grid_search.py into histograms

	env_base  	environment specified by environments.py
	root_me1 	baseline to compare to (2 - 1)/1
	root_me2	model of interest

	Results saved in model of interest, root_me2
	"""

	param_comb = params.get_params_all(root_me1)
	lmbdas = np.unique(np.array(param_comb)[:,3]) # list of all unique lmbds
	ks = params.get_ks(root_me1)[1:] # don't include leading 0.
	n_levels = 8 # number of levels of complexity

	plt.rcParams.update({'font.size': 8})
	fig_main, ax_main = plt.subplots(len(lmbdas),len(ks))
	fig_main.set_size_inches(22, 17)
	fig_main.tight_layout(pad=4.0, w_pad=4.0, h_pad=4.0)
	for k in ks:
		for lmbda in lmbdas:

			# track the mean diff in AUC between balanced/modulated
			# matched by parameters across levels
			mean_learn 	= np.zeros(n_levels)
			std_learn	= np.zeros(n_levels)
			mean_reward = np.zeros(n_levels)
			std_reward	= np.zeros(n_levels)


			save_path = "results/complexity_%s/%s/" %(root_me2,env_base)
			for level in np.arange(n_levels):
				# get environment name
				if level == 0:
					n_opt = ""              # no number for base environment
				else:   
					n_opt = str(level + 2)  # offset by 2
				env = env_base + n_opt

				path_base = "results/complexity_%s/%s/%s/k_%s/l_%.1f/" \
							%(root_me1,env_base,env,k,lmbda)
				path = "results/complexity_%s/%s/%s/k_%s/l_%.1f/" \
							%(root_me2,env_base,env,k,lmbda)


				# all param combinations
				auc_diff_learn = []
				auc_diff_reward = []

				auc_learn_modT = []
				auc_reward_modT = []
				auc_learn_modF = []
				auc_reward_modF = []

				# save parameters to color diff by AUC graphs
				color_beta = []
				color_alphaa = []
				titles = ["Beta","Alpha A"]
				saveas = ["diff_by_auc_beta","diff_by_auc_alphaa"]
				maps = ["plasma","plasma"]	# use coolwarm for boolean

				for par in param_comb:
					par2 = par[1:3]

					# handle parameter ish
					alpha_a, beta = par2
					color_alphaa.append(alpha_a)
					color_beta.append(beta)

					if os.path.exists(path + "mod_beta/params_" + str(par2) + ".pkle"):
						# get modulated data
						auc_learnT, auc_rewardT, learn_curve, reward_curve, rnd_seed = \
						pickle.load(open(path + "mod_beta/params_" + str(par2) + ".pkle","rb"))
						# get unmodulated data 
						auc_learnF, auc_rewardF, learn_curve, reward_curve, rnd_seed = \
						pickle.load(open(path_base + "mod_beta/params_" + str(par2) + ".pkle","rb"))

						# save perf difference, same parameters, modulated (T) or balanced (F)
						auc_diff_learn.append((auc_learnT - auc_learnF)/auc_learnF)
						auc_diff_reward.append((auc_rewardT - auc_rewardF)/auc_rewardF)

						auc_learn_modT.append(auc_learnT)
						auc_learn_modF.append(auc_learnF)
						auc_reward_modT.append(auc_rewardT)
						auc_reward_modF.append(auc_rewardF)

					else:
						tried = str(path + "mod_beta/params_" + str(par2) + ".pkle")
						print(tried)
						print("missing params", str(par2))
				colors = (color_beta,color_alphaa)

				# get mean and std for the level
				mean_learn[level] 	= np.mean(auc_diff_learn)
				std_learn[level]	= stats.sem(auc_diff_learn)
				mean_reward[level]	= np.mean(auc_diff_reward)
				std_reward[level]	= stats.sem(auc_diff_reward)

				# Graph AUC for each level
				what = "mod"
				# aucplotter.graph_learn(auc_learn_modT,auc_learn_modF,path,what)
				# aucplotter.graph_reward(auc_reward_modT,auc_reward_modF,path,what)
				# aucplotter.graph_diff(auc_diff_learn,path,"Learning",what)
				# aucplotter.graph_diff(auc_diff_reward,path,"Reward",what)
				# aucplotter.graph_diff_by_AUC(np.array(auc_learn_modF),np.array(auc_diff_learn),\
				# 	colors,titles,saveas,maps,\
				# 	path,"Learning",what)
				# aucplotter.graph_diff_by_AUC(np.array(auc_reward_modF),np.array(auc_diff_reward),\
				# 	colors,titles,saveas,maps,\
				# 	path,"Reward",what)

			# ext = "k%s_l%s_" %(int(k),int(lmbda))
			# # plot and save
			# xaxis = np.arange(n_levels)
			# plt.rcParams.update({'font.size': 22})
			# fig, ax = plt.subplots()
			# ax.errorbar(xaxis,mean_learn,yerr=std_learn)
			# plt.ylabel("AUC (Mod - Bal)/Bal")
			# plt.xlabel("Complexity")
			# plt.title("Learning")
			# plt.tight_layout()
			# plt.savefig(save_path + ext + "learning")
			# plt.close()

			# xaxis = np.arange(n_levels)
			# plt.rcParams.update({'font.size': 22})
			# fig, ax = plt.subplots()
			# ax.errorbar(xaxis,mean_learn,yerr=std_learn)
			# plt.ylabel("AUC (Mod - Bal)/Bal")
			# plt.xlabel("Complexity")
			# plt.title("Reward")
			# plt.tight_layout()
			# plt.savefig(save_path + ext + "reward")
			# plt.close()


			which_l = np.where(lmbdas == lmbda)[0][0]
			which_k = np.where(ks == k)[0][0]

			## add to the giant one
			xaxis = np.arange(n_levels)
			ax_main[which_k].errorbar(xaxis,mean_learn,yerr=std_learn)
			# ax_main[which_l, which_k].set_ylabel("AUC (Rho - Beta)/Beta")
			# ax_main[which_l, which_k].set_ylabel("Complexity")
			ax_main[which_k].set_title("k=%d l=%.2f" %(k,lmbda))

		print("k %d done" %(k))

	plt.savefig(save_path + "learning_cmp_%s_%s" %(root_me1,root_me2), dpi = 400)
	plt.close()



if __name__ == '__main__':
	main(sys.argv[1],sys.argv[2],sys.argv[3])





