##################################################################
# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
#
# Decription: Compare histogram of AUCs of model performance
# across the specified models
#
# Name: model_cmp/hist_cmp.py
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

def get_full_path(model,env,level,k,par,paths):
	"""
	model = key for dict paths
	env - rich/lean
	level - level of complexity
	par - parameters
	
	"""

	if level == 0:
		n_opt = ""              # no number for base environment
	else:   
		n_opt = "_" + str(level + 2)  # offset by 2
	path = paths[model]
	path = path  + env + "/" + env + n_opt + "/"
	ext = "params_" + str(par) + ".pkle"

	if not (model == "rl"):
		path_bal = path + "k_0.0/mod_constant/" 
		if model == "alu":
			path_mod = path + "k_%s/mod_value/" %(k)
		else:
			path_mod = path + "k_%s/mod_beta/" %(k)

		path_bal = path_bal + ext
		path_mod = path_mod + ext
		return(path_bal,path_mod)
	else:
		path = path + ext
		return path

def hist_AUC(sim1,ax,sim2=None):
	""" 
		Histogram of AUC under reward curve
		example sim1 - mod true
				sim2 - mod false
		what - which comparison is being done
			 - see function get_settings()
	"""

	plt.rcParams.update({'font.size': 30})
	ax.hist(sim1,color = (64/255, 64/255, 64/255, 0.5)) #gray
	if sim2 is not None:
		ax.hist(sim2,color = (137/255, 0, 1, 0.5)) #purple
	plt.ylabel("frequency")
	plt.xlabel("AUC")

def main(k,n_trials,n_states,start_plt,end_plt):
	
	n_levels = 8 # number of levels of complexity

	# model paths
	envs = ["rich","lean"]
	models = ["opal+","alu","rl"]
	append = "_%d_%d/" %(n_trials,n_states)
	paths = {"opal+": "../opal/sims/results/anneal_100/complexity" + append,\
			 "alu": "../bogacz/results/complexity" + append,\
			 "rl": "../opal/standard_rl/results/complexity" + append}

	# rl + opal had same grid
	par_key = "extrange"
	param_comb = params.get_params_all(par_key)
	param_alu = params.get_params_all("ALU2")
	param_dict = {"alu": param_alu, "rl": param_comb, "opal+": param_comb}

	#create directory for complex graphs when non-existing
	save_path = "trials%d_%d/k%s/" %(start_plt,end_plt,int(k))
	os.makedirs(save_path, exist_ok=True)
	fnt_sz = 25


	# TODO: limit high alpha in norm case
	for level in np.arange(n_levels):

		plt.rcParams.update({'font.size': fnt_sz})
		fig, axes = plt.subplots(len(models),len(envs))
		fig.set_size_inches(15, 20)  #width,height. # TODO adjust this
		fig.tight_layout(pad=4.0, w_pad=4.0, h_pad=4.0)

		for e_idx,env in enumerate(envs):
			print("environment: %s"%env)
			for m_idx,model in enumerate(models):

				# save the AUcs for all params
				if not (model == "rl"):
					auc_learn_all_mod = []
					auc_learn_all_bal = []
					auc_reward_all_mod = []
					auc_reward_all_bal = []
				else:
					auc_learn_all = []
					auc_reward_all = []

				for par in param_dict[model]:

					# get params formatted correctly
					if model == "alu":
						par2 = str(np.round(par,5))
					else:
						par2 = par[1:3]

					# non-RL models have DA mod and therefore
					# an extra level to their directory
					if not (model == "rl"):
						path_bal, path_mod = \
							get_full_path(model,env,level,k,par2,paths)

						# get modulated data 
						try:
							_, _, learn_curve_mod, reward_curve_mod, rnd_seed = \
							pickle.load(open(path_mod,"rb"))
						except: #ALU didn't save random seed oops
							_, _, learn_curve_mod, reward_curve_mod = \
							pickle.load(open(path_mod,"rb"))
						auc_learn_mod, auc_reward_mod = getAUC(learn_curve_mod,\
							reward_curve_mod,start_plt,end_plt)
						if model == "alu":
							print(auc_learn_mod)

						# get balanced data 
						try:
							_, _, learn_curve_bal, reward_curve_bal, rnd_seed = \
							pickle.load(open(path_bal,"rb"))
						except:
							_, _, learn_curve_bal, reward_curve_bal = \
							pickle.load(open(path_bal,"rb"))
						auc_learn_bal, auc_reward_bal = getAUC(learn_curve_bal,\
							reward_curve_bal,start_plt,end_plt)

						auc_learn_all_mod.append(auc_learn_mod)
						auc_learn_all_bal.append(auc_learn_bal)
						auc_reward_all_mod.append(auc_reward_mod)
						auc_reward_all_bal.append(auc_reward_bal)
					else:
						path = get_full_path(model,env,level,k,par2,paths)

						# get data
						_, _, learn_curve, reward_curve, rnd_seed = \
						pickle.load(open(path,"rb"))
						auc_learn, auc_reward = getAUC(learn_curve,\
							reward_curve,start_plt,end_plt)

						auc_learn_all.append(auc_learn)
						auc_reward_all.append(auc_reward)

				# plot the model data in the correct env
				this_ax = axes[m_idx,e_idx]
				if not (model == "rl"): 
					hist_AUC(auc_learn_all_bal,this_ax,sim2=auc_learn_all_mod)
				else:
					hist_AUC(auc_learn_all,this_ax)
				this_ax.set_ylabel("frequency",fontsize = fnt_sz)
				this_ax.set_xlabel("AUC",fontsize = fnt_sz)
				this_ax.set_title(model,fontsize = fnt_sz)

		# adjust the xlim so they are all the same
		# for ease of comparison
		# iteraxes = axes.reshape(-1)
		# min_x = 10^6
		# max_x = -10^6
		# for ax in iteraxes:
		# 	min_lim,max_lim = ax.get_xlim()
		# 	if min_lim < min_x:
		# 		mix_x = min_lim
		# 	if max_lim > max_x:
		# 		max_x = max_lim

		# new_lim = [min_x,max_x]
		# for ax in iteraxes:
		# 	ax.set_xlim(new_lim)	

		# same with ylims
		# min_y = 0
		# max_y = 0
		# for ax in iteraxes:
		# 	min_lim,max_lim = ax.get_ylim()
		# 	if max_lim > max_y:
		# 		max_y = max_lim

		# new_lim = [min_y,max_y]
		# for ax in iteraxes:
		# 	ax.set_ylim(new_lim)

		# save all that work!
		plt.rcParams.update({'font.size': fnt_sz})
		plt.savefig(save_path + "leaning_noptions%d" %(level+2), dpi = 400)
		plt.close()


if __name__ == '__main__':
    main(float(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]))