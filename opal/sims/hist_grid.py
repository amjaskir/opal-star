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
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import plot_dynamics

def graph_learn(mod_true,mod_false,path,ext):

	plt.rcParams.update({'font.size': 22})
	fig, ax = plt.subplots()
	ax.hist(mod_true,color = (137/255, 0, 1, 0.5))
	ax.hist(mod_false,color = (64/255, 64/255, 64/255, 0.5))
	plt.ylabel("frequency")
	plt.xlabel("AUC")
	#plt.title("Performance Curves")
	plt.tight_layout()
	plt.savefig(path + "learnhist_" + ext)
	plt.close()

def graph_learn_curve(states_mod,states_unmod,path,par,env):

	fig, ax = plt.subplots(figsize=(6, 5))
	plot_dynamics.avg_sm(states_unmod,40,1000,0,ax,"black")
	plot_dynamics.avg_sm(states_mod,40,1000,0,ax,"purple")
	plt.legend(("$\\rho=0$", "$\\rho \propto V$"), fontsize = 30, markerscale = 50, loc = 'lower right')
	plt.title(env + " Params:" + str(par))
	plt.xlabel("Trial")
	plt.ylabel("$\overline{p}$(best)")
	plt.tight_layout()
	plt.savefig(path + "learncurve")
	plt.close()

def graph_reward(mod_true,mod_false,path,ext):

	plt.rcParams.update({'font.size': 22})
	fig, ax = plt.subplots(figsize=(6, 5))
	ax.hist(mod_true,color = (137/255, 0, 1, 0.5))
	ax.hist(mod_false,color = (64/255, 64/255, 64/255, 0.5))
	plt.ylabel("frequency")
	plt.xlabel("AUC")
	plt.title("Reward Curves")
	plt.tight_layout()
	plt.savefig(path + "rewardhist_" + ext)
	plt.close()

def graph_diff(diff,path,ext):

	plt.rcParams.update({'font.size': 22})
	fig, ax = plt.subplots(figsize=(6, 5))
	ax.hist(diff,color = (137/255, 0, 1, 0.5))
	plt.ylabel("frequency")
	plt.xlabel("Online - Balanced AUC")
	#plt.title("Learning Curves")
	plt.tight_layout()
	plt.savefig(path + "diff_" + ext)
	plt.close()

def graph_diff_by_AUC(auc,diff,colors,path,ext):

	# get my colors
	color_beta,color_alphac,\
	color_alphaa,color_alpha_diff,\
	color_alphaa_greater,color_orig = colors

	# sort_idx = np.argsort(auc)	# don't really need to sort unless binning
	# sorted_auc = auc[sort_idx]
	# sorted_diff = diff[sort_idx]
	cor, p = stats.spearmanr(auc, diff)

	# horizontal line at zero info
	xs = np.linspace(np.min(auc),np.max(auc),200)
	h_line = np.array([0 for i in np.arange(len(xs))])
	plt.plot(xs, h_line, 'lightgray', linewidth=1) 

	s=5 # marker size
	# make origin params larger
	idx = np.where(np.array(color_orig) == 1)[0]
	sizes = np.zeros(len(auc)) + s
	sizes[idx] = 50
	plt.rcParams.update({'font.size': 22})

	# scatter color, highlighting original params
	fig, ax = plt.subplots(figsize=(6, 5))
	plt.plot(xs, h_line, 'lightgray', linewidth=1) 
	mapp = ax.scatter(auc,diff,c = np.array(color_orig),cmap = "coolwarm",s=sizes,zorder=10)
	#plt.colorbar(mapp)
	plt.ylabel("Online - Balanced AUC")
	plt.xlabel("Balanced AUC")
	plt.tight_layout()
	plt.savefig(path + "diff_by_auc_" + ext)
	plt.title("rho: %f p-val: %f" %(cor,p))
	plt.savefig(path + "diff_by_auc_withcorr_" + ext)
	plt.close()

	# by beta
	fig, ax = plt.subplots()
	plt.plot(xs, h_line, 'lightgray', linewidth=1) 
	mapp = ax.scatter(auc,diff,c = np.array(color_beta),cmap = "plasma",s=sizes,zorder=10)
	plt.colorbar(mapp)
	plt.ylabel("Online - Balanced AUC")
	plt.xlabel("Balanced AUC")
	plt.title("Beta")
	plt.tight_layout()
	plt.savefig(path + "diff_by_auc_beta_" + ext)
	plt.close()

	# by alphac
	fig, ax = plt.subplots()
	plt.plot(xs, h_line, 'lightgray', linewidth=1) 
	mapp = ax.scatter(auc,diff,c = np.array(color_alphac),cmap = "plasma",s=sizes,zorder=10)
	plt.colorbar(mapp)
	plt.ylabel("Online - Balanced AUC")
	plt.xlabel("Balanced AUC")
	plt.title("Alpha C")
	plt.tight_layout()
	plt.savefig(path + "diff_by_auc_alphac_" + ext)
	plt.close()

	# by alphaa
	fig, ax = plt.subplots()
	plt.plot(xs, h_line, 'lightgray', linewidth=1) 
	mapp = ax.scatter(auc,diff,c = np.array(color_alphaa),cmap = "plasma",s=sizes,zorder=10)
	plt.colorbar(mapp)
	plt.ylabel("Online - Balanced AUC")
	plt.xlabel("Balanced AUC")
	plt.title("Alpha A")
	plt.tight_layout()
	plt.savefig(path + "diff_by_auc_alphaa_" + ext)
	plt.close()

	# by alphac greater
	fig, ax = plt.subplots(figsize=(6, 5))
	plt.plot(xs, h_line, 'lightgray', linewidth=1) 
	mapp = ax.scatter(auc,diff,c = np.array(color_alphaa_greater),cmap = "coolwarm",s=sizes,zorder=10)
	#plt.colorbar(mapp)
	plt.ylabel("Online - Balanced AUC")
	plt.xlabel("Balanced AUC")
	plt.title("Alpha A > Alpha C")
	plt.tight_layout()
	plt.savefig(path + "diff_by_auc_alphaa_greater_" + ext)
	plt.close()

	# by alpha diff
	fig, ax = plt.subplots()
	plt.plot(xs, h_line, 'lightgray', linewidth=1) 
	mapp = ax.scatter(auc,diff,c = np.array(color_alpha_diff),cmap = "plasma",s=sizes,zorder=10)
	plt.colorbar(mapp)
	plt.ylabel("Online - Balanced AUC")
	plt.xlabel("Balanced AUC")
	plt.title("Alpha A - Alpha C")
	plt.tight_layout()
	plt.savefig(path + "diff_by_auc_alphadiff_" + ext)
	plt.close()


def main(env,norm,k,reprocess,ext):
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

	# which k to use
	k = float(k)
	path_base = "results/grid/%s/norm_%s/k_0.0/mod_constant/" \
				%(env,str(norm))
	path = "results/grid/%s/norm_%s/k_%s/" \
				%(env,str(norm),k)

	# check if cached data exists
	dump_it = path+ "saved_" + ext + ".pkle"
	if os.path.exists(dump_it) and not reprocess:
		auc_learn_modT,auc_learn_modF,auc_reward_modT,auc_reward_modF,\
		auc_diff_learn,auc_diff_reward = \
		pickle.load(open(dump_it,"rb"))

	else:
		# all param combinations
		if ext == "all":
			# alpha_cs = np.round(np.arange(.1,1.1,.1),1)
			# alpha_as = np.round(np.arange(.1,1.1,.1),1)
			# betas = np.round(np.concatenate([np.arange(.5,5.5,.5),\
			# 	np.arange(10,20.1,5)]),1)
			alpha_cs = np.round(np.arange(.1,1.1,.1),1)
			alpha_as = np.round(np.arange(.1,1.1,.1),1)
			betas = np.round(np.arange(.25,2.1,.25),2)
			param_comb = list(itertools.product(*[alpha_cs,alpha_as,betas]))
		elif ext == "orig": 	# original parametes
			alpha_cs = np.round(np.array([.1]),1)
			alpha_as = np.round(np.array([.3]),1)
			betas = np.round(np.array([1.5]),1)
			param_comb = list(itertools.product(*[alpha_cs,alpha_as,betas]))
		else:
			err = 'Invalid value given for arg ext. \"%s\" given' %exto
			raise Exception(err)

		auc_learn_modT = []
		auc_reward_modT = []
		auc_learn_modF = []
		auc_reward_modF = []
		auc_diff_learn = []
		auc_diff_reward = []

		# save parameters to color diff by AUC graphs
		color_beta = []
		color_alphac = []
		color_alphaa = []
		color_alpha_diff = []
		color_alphaa_greater = [] 	# T/F
		color_orig = []				# T/F

		for par in param_comb:
			#print('params: %s' % str(par))
			#sys.stdout.flush()

			# handle parameter ish
			alpha_c, alpha_a, beta = par
			color_beta.append(beta)
			color_alphac.append(alpha_c)
			color_alphaa.append(alpha_a)
			color_alpha_diff.append(alpha_a - alpha_c)
			color_alphaa_greater.append(int(alpha_a > alpha_c))
			if (alpha_c == .1) and (alpha_a == .3) and (beta == 1.5):
				color_orig.append(1)
			else:
				color_orig.append(0)



			if os.path.exists(path + "mod_value/params_" + str(par) + ".pkle"):
				# get modulated data
				auc_learnT, auc_rewardT, learn_curve, reward_curve = \
				pickle.load(open(path + "mod_value/params_" + str(par) + ".pkle","rb"))
				# get unmodulated data 
				auc_learnF, auc_rewardF, learn_curve, reward_curve = \
				pickle.load(open(path_base + "params_" + str(par) + ".pkle","rb"))

				# plot the learning curve if only looking at original params
				if ext == "orig":
					graph_learn_curve(states_mod,states_unmod,path,par,env)

				# save
				auc_learn_modT.append(auc_learnT)
				auc_learn_modF.append(auc_learnF)
				auc_reward_modT.append(auc_rewardT)
				auc_reward_modF.append(auc_rewardF)
				auc_diff_learn.append(auc_learnT - auc_learnF)
				auc_diff_reward.append(auc_rewardT - auc_rewardF)

			else:
				print("missing params" + par)
		colors = (color_beta,color_alphac,color_alphaa,color_alpha_diff,\
			color_alphaa_greater,color_orig)

		# save to regreaph later
		pickle.dump((auc_learn_modT,auc_learn_modF,auc_reward_modT,auc_reward_modF,\
			auc_diff_learn,auc_diff_reward,colors), \
			open(dump_it, "wb"))

	graph_learn(auc_learn_modT,auc_learn_modF,path,ext)
	graph_reward(auc_reward_modT,auc_reward_modF,path,ext)
	graph_diff(auc_diff_learn,path,ext)

	if ext == "all":
		graph_diff_by_AUC(np.array(auc_learn_modF),np.array(auc_diff_learn),colors,path,ext)

if __name__ == '__main__':
	main(sys.argv[1],bool(int(sys.argv[2])),int(sys.argv[3]),bool(int(sys.argv[4])),sys.argv[5])





