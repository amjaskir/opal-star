##################################################################
# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
#
# Decription: Compares learning of two variations of
# opal bayes simulations, for example one with standard
# value modulation and with learning rate modulation.
# Arguments to specify number of trials to plot.
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


def main(env_base,n_trials,n_states,pltn,variant="bmod"):
	"""
	Graphs data outputed by grid_search.py into histograms

	env_base  	environment specified by environments.py
	n_trials, n_states	specifies simulation
	pltn		number of trials to plot
	variant		variant to compare to, "bmod", "nohebb"
	"""

	# model versions for comparison
	# v1 - baseline/control
	# v2 - target model
	# baseline to compare to (2 - 1)/1
	v1 = "mod_beta/%s/" %(variant)
	v2 = "mod_beta/"


	#######################################################
	# This is copied from complexity.py
	par_key = "simplemod"
	param_comb = params.get_params_all(par_key)
	alpha_as = np.unique(np.array(param_comb)[:,1])[::2] # list of all unique alpha_as
	betas = np.unique(np.array(param_comb)[:,2])[::2]	# list of all unique betas
	print(alpha_as)
	print(betas)
	ks = params.get_ks(par_key)[1:] # don't include leading zero
	n_levels = 6 #9 # number of levels of complexity

	root_me = "simplemod/"
	mod = "mod_beta"

	# constant parameters for learning
	crit = "Bayes-SA"
	root_me = root_me + crit + "/"

	# reward mag and loss mag
	# same for each option
	r_mag = 1
	l_mag = 0
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

	plt.rcParams.update({'font.size': 8})
	fig_main, ax_main = plt.subplots(1,len(ks))
	fig_main.set_size_inches(22, 17)
	fig_main.tight_layout(pad=4.0, w_pad=4.0, h_pad=4.0)


	#create directory for complex graphs when non-existing
	# save in model of interest directory under new file "variant"
	save_path = "results/%s/trials%d_sims%d/%s/ntrials%d/%s/cmp_%s/"\
			%(root_me,n_trials,n_states,env_base,pltn,mod,variant)
	os.makedirs(save_path, exist_ok=True)

	for k in ks:
		# track the mean diff in AUC between balanced/modulated
		# matched by parameters across levels
		mean_learn 	= np.zeros(n_levels)
		std_learn	= np.zeros(n_levels)
		mean_reward = np.zeros(n_levels)
		std_reward	= np.zeros(n_levels)

		for level in np.arange(n_levels):
			env = "%s_%d_%d" %(env_base,10,level+2)

			bookmark1 = "results/%s/trials%d_sims%d/%s/k_%s/" %(root_me,n_trials,n_states,env,k)
			bookmark2 = "results/%s/trials%d_sims%d/%s/k_%s/" %(root_me,n_trials,n_states,env,k)
			path_v1 = bookmark1 + v1
			path_v2 = bookmark2 + v2

			# save in separate folder for the specified pltn
			save_pltn = bookmark2 + "ntrials" + str(pltn) + "/" + "cmp_%s/" %(variant)
			os.makedirs(save_pltn, exist_ok=True)

			auc_diff_learn = []
			auc_diff_reward = []

			auc_learn_modT = []		#v2, target model
			auc_reward_modT = []	
			auc_learn_modC = []		#v1, baseline/control model
			auc_reward_modC = []

			# save parameters to color diff by AUC graphs
			color_beta = []
			color_alphaa = []
			titles = ["Beta","Alpha A"]
			saveas = ["diff_by_auc_beta","diff_by_auc_alphaa"]
			maps = ["plasma","plasma"]	# use coolwarm for boolean

			plt.rcParams.update({'font.size': 8})
			fig_curv, ax_curv = plt.subplots(len(alpha_as),len(betas))
			fig_curv.set_size_inches(22, 17)
			fig_curv.tight_layout(pad=4.0, w_pad=4.0, h_pad=4.0)
			for par in param_comb:
				# handle parameter ish
				par2 = par[1:3]
				alpha_a, beta = par2

				# if I'm skipping anything
				if alpha_a not in alpha_as:
					continue
				if beta not in betas:
					continue

				color_alphaa.append(alpha_a)
				color_beta.append(beta)

				if os.path.exists(path_v1 + "/params_" + str(par2) + ".pkle"):
					# get baseline model 
					_, _, learn_curveC, reward_curveC, _ = \
					pickle.load(open(path_v1 + "/params_" + str(par2) + ".pkle","rb"))
					auc_learnC, auc_rewardC = getAUC(learn_curveC,reward_curveC,pltn)

					# get target model (likely modulated model)
					_, _, learn_curveT, reward_curveT, _ = \
					pickle.load(open(path_v2 + "/params_" + str(par2) + ".pkle","rb"))
					auc_learnT, auc_rewardT = getAUC(learn_curveT,reward_curveT,pltn)

					# save perf difference, same parameters, modulated (T) or balanced (F)
					auc_diff_learn.append((auc_learnT - auc_learnC)/auc_learnC)
					auc_diff_reward.append((auc_rewardT - auc_rewardC)/auc_rewardC)

					auc_learn_modT.append(auc_learnT)
					auc_learn_modC.append(auc_learnC)
					auc_reward_modT.append(auc_rewardT)
					auc_reward_modC.append(auc_rewardC)

					# save learning curves for parameter
					aucplotter.learn_curve(learn_curveT,auc_learnT,learn_curveC,auc_learnC,par2,save_pltn,\
						w_auc=True,pltn=pltn)

					# save learning curves in larger grid figure
					which_a = np.where(alpha_as == alpha_a)[0][0]
					which_b = np.where(betas == beta)[0][0]
					this_ax = ax_curv[which_a,which_b]
					aucplotter.learn_curve(learn_curveT,auc_learnT,learn_curveC,auc_learnC,par2,save_pltn,\
						w_auc=True,pltn=pltn,ax=this_ax)

				else:
					tried = str(path_v1 + "/params_" + str(par2) + ".pkle")
					print(tried)
					print("missing params", str(par2))

			# save larger curve grid
			plt.rcParams.update({'font.size': 8})
			plt.savefig(save_pltn + "allcurves.png",dpi = 500)
			plt.close()

			colors = (color_beta,color_alphaa)

			# get mean and std for the level
			mean_learn[level] 	= np.mean(auc_diff_learn)
			std_learn[level]	= stats.sem(auc_diff_learn)
			mean_reward[level]	= np.mean(auc_diff_reward)
			std_reward[level]	= stats.sem(auc_diff_reward)

			# Graph AUC for each level
			# what = "mod"
			what = variant
			aucplotter.graph_learn(auc_learn_modT,auc_learn_modC,save_pltn,what)
			aucplotter.graph_reward(auc_reward_modT,auc_reward_modC,save_pltn,what)
			aucplotter.graph_diff(auc_diff_learn,save_pltn,"Learning",what)
			aucplotter.graph_diff(auc_diff_reward,save_pltn,"Reward",what)
			aucplotter.graph_diff_by_AUC(np.array(auc_learn_modC),np.array(auc_diff_learn),\
				colors,titles,saveas,maps,\
				save_pltn,"Learning",what)
			aucplotter.graph_diff_by_AUC(np.array(auc_reward_modC),np.array(auc_diff_reward),\
				colors,titles,saveas,maps,\
				save_pltn,"Reward",what)

			aucplotter.auc_by_alpha(np.array(auc_learn_modC),color_alphaa,save_pltn,"Learning")
			aucplotter.auc_by_alpha(np.array(auc_reward_modC),color_alphaa,save_pltn,"Reward")

			ext = "k%s" %(int(k))

		# plot and save
		# for learning
		xaxis = np.arange(n_levels)
		plt.rcParams.update({'font.size': 22})
		fig, ax = plt.subplots()
		ax.errorbar(xaxis,mean_learn,yerr=std_learn)
		plt.ylabel("AUC (DA - %s)/%s" %(variant,variant))
		plt.xlabel("Complexity")
		plt.title("Learning")
		plt.tight_layout()
		plt.savefig(save_path + ext + "learning")
		plt.close()

		# for reward
		xaxis = np.arange(n_levels)
		plt.rcParams.update({'font.size': 22})
		fig, ax = plt.subplots()
		ax.errorbar(xaxis,mean_learn,yerr=std_learn)
		plt.ylabel("AUC (DA - %s)/%s" %(variant,variant))
		plt.xlabel("Complexity")
		plt.title("Reward")
		plt.tight_layout()
		plt.savefig(save_path + ext + "reward")
		plt.close()


		## add to the giant one
		which_k = np.where(ks == k)[0][0]
		xaxis = np.arange(n_levels)
		ax_main[which_k].errorbar(xaxis,mean_learn,yerr=std_learn)
		# ax_main[which_l, which_k].set_ylabel("AUC (Mod - Bal)/Bal")
		# ax_main[which_l, which_k].set_ylabel("Complexity")
		ax_main[which_k].set_title("k=%d" %k)

		print("k %d done" %(k))

	plt.rcParams.update({'font.size': 22})
	plt.savefig(save_path + "learning_cmp_%s" %(variant), dpi = 400)
	plt.close()


if __name__ == '__main__':
	if len(sys.argv) < 6:
		main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))
	else:
		# use a variant of the opal code
		main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),sys.argv[5])





