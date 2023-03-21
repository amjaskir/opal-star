##################################################################
# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
#
# Decription: Compare effects of DA modulation in ALU model vs
# the OpAL model in designated trial range.
# Specifically, understand asympotptic behavior without modulation
# Name: bogacz/asymp_analysis.py
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

def getAUC(learn_curve,reward_curve,start_plt,end_plt):
	[avg_sm,sem_sm] = learn_curve
	[avg_r,sem_r] = reward_curve
	n_trials = end_plt - start_plt

	# calc AUC of both
	# make end plt inclusive
	x = np.arange(n_trials)
	auc_learn = metrics.auc(x,avg_sm[start_plt:end_plt])
	auc_reward = metrics.auc(x,avg_r[start_plt:end_plt])

	return auc_learn, auc_reward

def main(n_trials,n_states,start_plt,end_plt):
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
	envs = ["rich","lean"]
	alu_param_comb = params.get_params_all("ALU2")
	opal_param_comb = params.get_params_all("extrange")

	# restrict alpha actor for opal? alu?
	restrict = True
	athresh_opal = .8
	athresh_alu = .5

	# save in bogacz directory,  make dir if doesn't yet exist
	save_path = "./results/asymp/"
	os.makedirs(save_path, exist_ok=True)

	# prepare plots ---------------------------
	plt.rcParams.update({'font.size': 18})

	# hist
	fig, axs = plt.subplots(2,1)
	ax_learn_A = axs[0]
	ax_learn_O = axs[1]
	min_auc = 10^6
	max_auc = 0

	# bar
	fig_bar, ax_bar = plt.subplots()

	# go through each env ----------------------------------------------
	for env in envs:

		# ALU params + paths
		alu_path = "./results/%s_%d_%d/k_0.0/" %(env,n_trials,n_states)

		# OpAL params + paths
		full = True
		c_only = True
		if full:
			if c_only:
				opal_path = "../opal/sims/results/full/complexity_%d_%d/%s/%s/k_0.0/" \
					%(n_trials,n_states,env,env)
			else:
				opal_path = "../opal/sims/results/full_c_only/complexity_%d_%d/%s/%s/k_0.0/" \
					%(n_trials,n_states,env,env)
		else:
			opal_path = "../opal/sims/results/complexity_%d_%d/%s/%s/k_0.0/" \
					%(n_trials,n_states,env,env)
		print(opal_path)

		# ALU ------------------------------------------------
		# initialize arrays 
		alu_auc_learn = []
		alu_auc_reward = []
		alu_performance = []	# trial avg
		for par in alu_param_comb:
			r_par = np.round(par,5)

			# handle parameter ish
			alpha_a, epsilon, lbda, beta = r_par

			# restrict considered parameters to alpha's 
			# in specified range
			if restrict and (alpha_a >= athresh_alu):
				# print("ignored alpha %f" %alpha_a)
				continue

			if os.path.exists(alu_path + "mod_constant/params_" + str(r_par) + ".pkle"):
				# get data
				_, _, learn_curve, reward_curve = \
				pickle.load(open(alu_path + "mod_constant/params_" + str(r_par) + ".pkle","rb"))
				auc_learn, auc_reward = getAUC(learn_curve,reward_curve,start_plt,end_plt)
				[avg_sm,_] = learn_curve

				# save
				alu_auc_learn.append(auc_learn)
				alu_auc_reward.append(auc_reward)
				alu_performance.append(np.mean(avg_sm[start_plt:end_plt]))
			else:
				tried = str(alu_path + "mod_constant/params_" + str(r_par) + ".pkle")
				print(tried)
				print("missing params", str(r_par))


		# OpAL ------------------------------------------------
		opal_auc_learn = []
		opal_auc_reward = []
		opal_performance = []
		for par in opal_param_comb:	
			par2 = par[1:3]
			alpha_a, beta = par2

			# restrict considered parameters to alpha's 
			# in specified range
			if restrict and (alpha_a >= athresh_opal):
				# print("ignored alpha %f" %alpha_a)
				continue

			if os.path.exists(opal_path + "mod_constant/params_" + str(par2) + ".pkle"):
				_, _, learn_curve, reward_curve, _ = \
				pickle.load(open(opal_path + "mod_constant/params_" + str(par2) + ".pkle","rb"))
				auc_learn, auc_reward = getAUC(learn_curve,reward_curve,start_plt,end_plt)
				[avg_sm,_] = learn_curve

				# save
				opal_auc_learn.append(auc_learn)
				opal_auc_reward.append(auc_reward)
				opal_performance.append(np.mean(avg_sm[start_plt:end_plt]))
			else:
				tried = str(opal_path + "mod_constant/params_" + str(par2) + ".pkle")
				print(tried)
				print("missing params", str(par2))

		# plot for env ------------------------------------------

		# rich colors
		cr_a = (0, 125/255, 0, 0.5) #dark green, rich env
		cr_o = (0, 204/255, 0, 0.5) #light green, rich env
		# lean colors
		cl_a = (64/255, 64/255, 64/255, 0.5) #dark grey, lean env
		cl_o = (170/255, 170/255, 170/255, 0.5) #light grey, lean env

		# histogram
		if env == "rich":
			ax_learn_A.hist(alu_auc_learn, color = cr_o)
			ax_learn_O.hist(opal_auc_learn, color = cr_o)
		else:
			ax_learn_A.hist(alu_auc_learn,color = cl_o)
			ax_learn_O.hist(opal_auc_learn,color = cl_o)

		# get min and max aucs for normalizing
		if min(alu_auc_learn) < min_auc:
			min_auc = min(alu_auc_learn)
		if min(opal_auc_learn) < min_auc: 
			min_auc = min(opal_auc_learn)
		if max(alu_auc_learn) > max_auc:
			max_auc = max(alu_auc_learn)
		if max(opal_auc_learn) > max_auc: 
			max_auc = max(opal_auc_learn)

		# bar graphs of avg
		if env == "rich":
			ax_bar.bar(1,np.mean(alu_performance),yerr=stats.sem(alu_performance), color = cr_o)
			ax_bar.bar(4,np.mean(opal_performance),yerr=stats.sem(opal_performance), color = cr_o)
		else:
			ax_bar.bar(2,np.mean(alu_performance),yerr=stats.sem(alu_performance), color = cl_o)
			ax_bar.bar(5,np.mean(opal_performance),yerr=stats.sem(opal_performance), color = cl_o)

		# mark progress
		print("end env: %s" %env)

	# save things ------------------------------------------------------
	# save bar
	labels = ["RA","LA","RO","LO"]
	x = np.array([1,2,4,5])
	ax_bar.set_ylabel("avg performance")
	ax_bar.set_xticks(x)
	ax_bar.set_xticklabels(labels)
	ax_bar.set_ylim(0,1)
	plt.tight_layout()

	fname = "perf" + str(start_plt) + "_" + str(end_plt)
	fig_bar.savefig(save_path + fname)
	plt.close(fig_bar)

	# save the hists
	ax_learn_A.set_xlim(min_auc,max_auc)
	ax_learn_A.set_ylabel("frequency")
	ax_learn_A.set_xlabel("AUC")
	ax_learn_A.set_title("ALU")

	# save the hists
	ax_learn_O.set_xlim(min_auc,max_auc)
	ax_learn_O.set_ylabel("frequency")
	ax_learn_O.set_xlabel("AUC")
	ax_learn_O.set_title("OpAL")
	ax_learn_O.legend(["Rich","Lean"], fontsize = "small")
	plt.tight_layout()

	fname = "learnhist" + str(start_plt) + "_" + str(end_plt)
	fig.savefig(save_path + fname)
	plt.close(fig)

		

if __name__ == '__main__':
	main(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))





