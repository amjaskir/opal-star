##################################################################
# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
#
# Decription: Compare effects of DA modulation in bayesian opal model
#
# Name: bogacz/hist_grid_bayes.py
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

	root_me = "revisions/BOGACZ/"

	par_key = "ALU2"
	param_comb = params.get_params_all(par_key)
	ks = params.get_ks(par_key)[1:] # don't include leading 0.
	alpha_as = np.unique(np.array(param_comb)[:,0]) # list of all unique alpha_as
	epsilons = np.unique(np.array(param_comb)[:,1]) # list of all unique epsilons
	lmbdas = np.unique(np.array(param_comb)[:,2]) 	# list of all unique lmbds
	betas = np.unique(np.array(param_comb)[:,3]) 	# list of all unique betas
	

	n_levels = 5 # number of levels of complexity
	full = False

	# complexity graphs x ks
	plt.rcParams.update({'font.size': 8})
	fig_main, ax_main = plt.subplots(1,len(ks))
	fig_main.set_size_inches(22, 17)
	fig_main.tight_layout(pad=4.0, w_pad=4.0, h_pad=4.0)


	#create directory for complex graphs when non-existing
	save_path = "results/%s_%d_%d/%s/ntrials%d/"\
		%(root_me,n_trials,n_states,env_base,pltn)
	os.makedirs(save_path, exist_ok=True)

	# overall avg AUC for each k
	avg_auc_mod = np.zeros((len(ks),n_levels))
	sem_auc_mod = np.zeros((len(ks),n_levels))

	for k_idx,k in enumerate(ks):
		# track the mean diff in AUC between balanced/modulated
		# matched by parameters across levels
		mean_learn 	= np.zeros(n_levels)
		std_learn	= np.zeros(n_levels)
		mean_reward = np.zeros(n_levels)
		std_reward	= np.zeros(n_levels)

		for level in np.arange(n_levels):
			print("level: %s" %level)
			# get environment name
			env = "%s_%d_%d" %(env_base,10,level+2)

			if full:
				add_on = "_FULL"
			else:
				add_on = ""
			path_base = "results/%s_%d_%d%s/%s/%s/k_0.0/mod_constant/" \
				%(root_me,n_trials,n_states,add_on,env_base,env)
			path = "results/%s_%d_%d%s/%s/%s/k_%s/" \
				%(root_me,n_trials,n_states,add_on,env_base,env,k)
			save_pltn = path + "ntrials" + str(pltn) + "/"
			
			# create dir if doesn not exist
			os.makedirs(save_pltn, exist_ok=True)

			auc_learn_modT = []
			auc_learn_modF = []
			auc_reward_modT = []
			auc_reward_modF = []
			auc_diff_learn = []
			auc_diff_reward = []
			auc_abs_diff_learn = []
			auc_abs_diff_reward = []

			# save parameters to color diff by AUC graphs
			color_alpha = []
			color_epsilon = []
			color_lbda = []
			color_beta = [] 
			titles = ["Alpha A","Beta","Epsilon (Nonlinearity)","Lambda (Decay)"]
			saveas = ["diff_by_auc_alphaa","diff_by_auc_beta","diff_by_eps","diff_by_lmbda",]
			maps = ["plasma","plasma","plasma","plasma"]	# use coolwarm for boolean

			for par in param_comb:
				r_par = np.round(par,5)

				# handle parameter ish
				alpha_a, epsilon, lbda, beta = r_par
				color_alpha.append(alpha_a)
				color_epsilon.append(epsilon)
				color_lbda.append(lbda)
				color_beta.append(beta)

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
					auc_abs_diff_learn.append((auc_learnT - auc_learnF))
					auc_abs_diff_reward.append((auc_rewardT - auc_rewardF))

					# save learning curves for parameter
					aucplotter.learn_curve(learn_curveT,auc_learnT,learn_curveF,auc_learnF,r_par,save_pltn,\
						w_auc=True,pltn=pltn)

					# TODO: FIGURE OUT IF I WANT TO PLOT LEARNING CURVES IN LARGER GRID

				else:
					tried = str(path + "mod_value/params_" + str(r_par) + ".pkle")
					print(tried)
					print("missing params", str(r_par))

			colors = (color_alpha,color_beta,color_epsilon,color_lbda)

			# get mean and std of auc diff for the level
			mean_learn[level] 	= np.mean(auc_diff_learn)
			std_learn[level]	= stats.sem(auc_diff_learn)
			mean_reward[level]	= np.mean(auc_diff_reward)
			std_reward[level]	= stats.sem(auc_diff_reward)

			# save mean and std of mod auc perf
			avg_auc_mod[k_idx,level] = np.mean(auc_learn_modT)
			sem_auc_mod[k_idx,level] = stats.sem(auc_learn_modT)


			# Graph AUC for each level
			what = "mod"
			aucplotter.graph_learn(auc_learn_modT,auc_learn_modF,save_pltn,what)
			aucplotter.graph_reward(auc_reward_modT,auc_reward_modF,save_pltn,what)

			# difference in AUC
			aucplotter.graph_diff(auc_diff_learn,save_pltn,"Learning",what)
			aucplotter.graph_diff(auc_diff_reward,save_pltn,"Reward",what)
			aucplotter.graph_diff_by_AUC(np.array(auc_learn_modF),np.array(auc_diff_learn),\
				colors,titles,saveas,maps,\
				save_pltn,"Learning",what)
			aucplotter.graph_diff_by_AUC(np.array(auc_reward_modF),np.array(auc_diff_reward),\
				colors,titles,saveas,maps,\
				save_pltn,"Reward",what)

			# now for absolute difference
			aucplotter.graph_diff(auc_abs_diff_learn,save_pltn,"Learning",what,abs_diff=True)
			aucplotter.graph_diff(auc_abs_diff_reward,save_pltn,"Reward",what,abs_diff=True)
			aucplotter.graph_diff_by_AUC(np.array(auc_learn_modF),np.array(auc_abs_diff_learn),\
				colors,titles,saveas,maps,\
				save_pltn,"Learning",what,abs_diff=True)
			aucplotter.graph_diff_by_AUC(np.array(auc_reward_modF),np.array(auc_abs_diff_reward),\
				colors,titles,saveas,maps,\
				save_pltn,"Reward",what,abs_diff=True)

			# AUC according to alphas, high alpha should have decrease in AUC
			aucplotter.auc_by_alpha(np.array(auc_learn_modF),color_alpha,save_pltn,"Learning")
			aucplotter.auc_by_alpha(np.array(auc_reward_modF),color_alpha,save_pltn,"Reward")

		ext = "k%s" %(int(k))

		# plot and save
		# for learning
		xaxis = np.arange(n_levels)
		plt.rcParams.update({'font.size': 22})
		fig, ax = plt.subplots()
		ax.errorbar(xaxis,mean_learn,yerr=std_learn)
		plt.ylabel("AUC (Mod - Bal)/Bal")
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
		plt.ylabel("AUC (Mod - Bal)/Bal")
		plt.xlabel("Complexity")
		plt.title("Reward")
		plt.tight_layout()
		plt.savefig(save_path + ext + "reward")
		plt.close()

		## add to the giant one
		which_k = np.where(ks == k)[0][0]
		xaxis = np.arange(n_levels)
		ax_main[which_k].errorbar(xaxis,mean_learn,yerr=std_learn)
		ax_main[which_k].set_title("k=%d" %k)

		print("k %d done" %(k))

	plt.rcParams.update({'font.size': 22})
	plt.savefig(save_path + "learning", dpi = 400)
	plt.close()

	# plot avg auc by k and complexity level
	plt.rcParams.update({'font.size': 22})
	fig, ax = plt.subplots()
	plt.rcParams.update({'font.size': 22})
	xaxis = np.arange(n_levels)
	for k_idx,k in enumerate(ks):
		offset = k_idx/10
		plt.errorbar(xaxis+offset,avg_auc_mod[k_idx,:],yerr=sem_auc_mod[k_idx,:],label=str(k),linewidth=1.0)
	plt.ylabel("Avg AUC Mod")
	plt.xlabel("Complexity")
	plt.title("Learning")
	plt.savefig(save_path + "auc_by_k", dpi = 400)
	plt.close()



if __name__ == '__main__':
	main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))





