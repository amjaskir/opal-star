##################################################################
# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
#
# Decription: Compare histogram of AUCs of model performance
# across the specified models. Particularly for unmodulated models
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
import pylab
from sklearn import metrics

# get my helper modules
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../helpers'))
import params

def getAUC(learn_curve,reward_curve,start_plt,end_plt):
	start_plt = 0
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

	path = paths[model]
	full_env = "%s_%d_%d/" %(env,10,level+2)

	# correct some path things for ALU
	if model == "alu":
		path = path  + env + "/" + full_env
	elif model == "alu-1":
		if env == "30":
			dummyenv = "lean"
		else:
			dummyenv = "rich"
		full_env = "%s_%d/" %(dummyenv,level+2)
		path = path  + dummyenv + "/" + full_env
	else:
		path = path  + full_env
	ext = "params_" + str(par) + ".pkle"

	# account for k if needed
	if model == "alu" or model == "alu-1" or model == "OpAL":
		path_final = path + "k_0.0/mod_constant/"
	elif model == "RL" or model == "UCB":
		path_final = path
	elif model == "SACrit_BayesAnneal_mod" or model == "SACrit_BayesAnneal_mod_phi0.5" or model == "SACrit_BayesAnneal_mod_phi1.5":
		path_final = path + "k_%s/mod_beta/" %(int(k))
	elif model == "SACrit_BayesAnneal": #should handle the revision models
		path_final = path + "k_0/mod_constant/"
	elif model == "SAcrit":
		path_final = path + "k_%s/mod_beta/" %(k)
	elif model == "OpAL*":
		path_final = path + "k_%s/mod_beta/" %(k)
	elif model == "NoHebb":
		path_final = path + "k_%s/mod_beta/nohebb/" %(k)
		# path_final = path + "k_0/mod_constant/nohebb/" 
	else:
		path_final = path + "k_%s/mod_beta/bmod/" %(k)
		# path_final = path + "k_0/mod_constant/bmod/" 

	return path_final + ext

def hist_AUC(models,save_path,env,level,curve_paths,start_plt,end_plt,curve_params,normalize=False):
	""" 
		Histogram of AUC under reward curve
		example sim1 - mod true
				sim2 - mod false
		what - which comparison is being done
			 - see function get_settings()
		normalize - normalize by number of parameters
	"""

	fnt_sz = 10 #20
	plt.rcParams.update({'font.size': fnt_sz})
	fig, ax = plt.subplots(figsize=(6, 5))
	# fig.set_size_inches(15, 20)  #width,height.
	fig.tight_layout(pad=4.0, w_pad=4.0, h_pad=4.0)
	peak_curve = {}	# histogram peak
	peak_auc = {}
	max_curve = {}	# max hist value
	max_auc = {}
	max_params = {}

	print(models)
	for key in models:
		histme = models[key]
		if key == "UCB_opt":
			ax.axvline(x=histme)
			peak_path = 0
			max_path = 0
		else:
			histv, bins, _ = ax.hist(histme,alpha=0.5,density=normalize)
			ax.set_ylabel("frequency",fontsize = fnt_sz)
			ax.set_xlabel("AUC",fontsize = fnt_sz)

			# find the peak
			hist_peak_arr = np.where(histv == max(histv))[0]
			peak_middle = round(len(hist_peak_arr)/2)
			peak = hist_peak_arr[peak_middle]
			minAUC = bins[peak]
			maxAUC = bins[peak+1]

			# find a corresponding path in the bin
			# take the first one if more than 1
			peak_path = np.where((histme > minAUC) & (histme < maxAUC))[0][0] 

			# find the max path
			max_path = np.where(histme == np.max(histme))[0][0]

		try:
			try:
				_, _, learn_curve_peak, _, _ = \
							pickle.load(open(curve_paths[key][peak_path],"rb"))
				_, _, learn_curve_max, _, _ = \
							pickle.load(open(curve_paths[key][max_path],"rb"))
			except:
				print("MISSING PARAMS")
				continue
		except:
			try:
				_, _, learn_curve_peak, _ = \
							pickle.load(open(curve_paths[key][peak_path],"rb"))
				_, _, learn_curve_max, _ = \
							pickle.load(open(curve_paths[key][max_path],"rb"))
			except:
				print("MISSING PARAMS")
				continue
		peak_curve[key] = learn_curve_peak
		peak_auc[key] = histme[peak_path]
		max_curve[key] = learn_curve_max
		max_auc[key] = histme[max_path]
		max_params[key] = curve_params[key][max_path]

	if env == "30":
		plt.title("Lean Environment, %s options" %(level+2))
	else:
		plt.title("Rich Environment, %s options" %(level+2))
	plt.legend(models.keys())
	plt.savefig(save_path + "leaning_%s_%d" %(env,level+2), dpi = 400)
	plt.close()

	# plot the learning curves, peak
	fig, ax = plt.subplots()
	fig.set_size_inches(15, 20)  #width,height.
	fig.tight_layout(pad=4.0, w_pad=4.0, h_pad=4.0)
	for key in models:
		(avg_sm,sem_sm) = peak_curve[key]
		(avg_sm,sem_sm) = (avg_sm[start_plt:end_plt],sem_sm[start_plt:end_plt])
		xaxis = np.arange(len(avg_sm))
		ax.errorbar(xaxis,avg_sm,yerr=sem_sm,label=key + " AUC: " + str(peak_auc[key]),linewidth=2.)
	plt.legend()
	plt.savefig(save_path + "leaningPeakCurves_%s_%d" %(env,level+2), dpi = 400)
	plt.close()

	# again with no legend, peak
	fig, ax = plt.subplots()
	fig.set_size_inches(15, 20)  #width,height.
	fig.tight_layout(pad=4.0, w_pad=4.0, h_pad=4.0)
	for key in models:
		(avg_sm,sem_sm) = max_curve[key]
		(avg_sm,sem_sm) = (avg_sm[start_plt:end_plt],sem_sm[start_plt:end_plt])
		xaxis = np.arange(len(avg_sm))
		ax.errorbar(xaxis,avg_sm,yerr=sem_sm,linewidth=2.)
		pylab.fill_between(xaxis, avg_sm, alpha=0.25) # shade under
	plt.gca().set_ylim(bottom=0.5) # only works for 2 options
	plt.savefig(save_path + "leaningPeakCurves_nolegend_%s_%d" %(env,level+2), dpi = 400)
	plt.close()

	# plot the learning curves, max val
	fig, ax = plt.subplots()
	fig.set_size_inches(15, 20)  #width,height.
	fig.tight_layout(pad=4.0, w_pad=4.0, h_pad=4.0)
	for key in models:
		(avg_sm,sem_sm) = max_curve[key]
		(avg_sm,sem_sm) = (avg_sm[start_plt:end_plt],sem_sm[start_plt:end_plt])
		xaxis = np.arange(len(avg_sm))
		ax.errorbar(xaxis,avg_sm,yerr=sem_sm,label=key + "Params: " + str(max_params[key]),linewidth=2.)
	plt.legend()
	plt.savefig(save_path + "leaningMaxCurves_%s_%d" %(env,level+2), dpi = 400)
	plt.close()

	# again with no legends, max val
	fig, ax = plt.subplots()
	fig.set_size_inches(15, 20)  #width,height.
	fig.tight_layout(pad=4.0, w_pad=4.0, h_pad=4.0)
	for key in models:
		(avg_sm,sem_sm) = max_curve[key]
		(avg_sm,sem_sm) = (avg_sm[start_plt:end_plt],sem_sm[start_plt:end_plt])
		xaxis = np.arange(len(avg_sm))
		ax.errorbar(xaxis,avg_sm,yerr=sem_sm,linewidth=2.)
		pylab.fill_between(xaxis, avg_sm, alpha=0.25) # shade under
	plt.gca().set_ylim(bottom=0.5) # only works for 2 options
	plt.savefig(save_path + "leaningMaxCurves_nolegend_%s_%d" %(env,level+2), dpi = 400)
	plt.close()

	# to use for cross env
	return max_params

def swap_max_params(all_max_params,models,paths,level,k,start_plt,end_plt,save_path):
	
	envs = ["80","30"]
	# plot max params in one environment with the alternative
	for _,env in enumerate(envs):

		print("using env %s", env)
		if env == "80":
			alt_env = "30"
		else:
			alt_env = "80"
		
		origin_full_env = "%s_%d_%d" %(env,10,level+2)

		# plot things
		fig, ax = plt.subplots()
		fig.set_size_inches(15, 20)  #width,height.
		fig.tight_layout(pad=4.0, w_pad=4.0, h_pad=4.0)
		
		# plot by model
		for model in models:
			print("using model %s", model)

			# get the max params for origin
			max_params = all_max_params[origin_full_env][model]

			# get path for alt env with max params
			alt_path_final = get_full_path(model,alt_env,level,k,str(max_params),paths)

			_, _, learn_curve, _, _ = \
						pickle.load(open(alt_path_final,"rb"))

			# plot
			(avg_sm,sem_sm) = learn_curve
			(avg_sm,sem_sm) = (avg_sm[start_plt:end_plt],sem_sm[start_plt:end_plt])
			xaxis = np.arange(len(avg_sm))
			ax.errorbar(xaxis,avg_sm,yerr=sem_sm,label=model+ "Params: " + str(max_params),linewidth=2.)

		# save
		plt.legend()
		plt.savefig(save_path + "leaningMaxCurves_swap_%s_%d" %(alt_env,level+2), dpi = 400)
		plt.close()
		

def main(k,start_plt,end_plt):
	
	n_levels = 2 # number of levels of complexity

	# model paths
	envs = ["80","30"]
	# models = ["opal*","Balanced","NoHebb","alu","alu-1"]
	# models = ["opal*","Balanced","NoHebb","Bmod"]
	# models = ["OpAL*","OpAL","NoHebb"]
	models = ["Bmod", "SACrit_BayesAnneal", "SACrit_BayesAnneal_mod"]
	# models = ["OpAL", "BayesCrit_StdAnneal"]
	paths = {"OpAL*": "../opal/sims/results/simplemod/Bayes-SA/rmag_1_lmag_0_mag_1/phi_1.0/usestd_False/exp_val_False/anneal_True_T_10/trials500_sims5000/",\
			 "OpAL": "../opal/sims/results/simplemod/Bayes-SA/rmag_1_lmag_0_mag_1/phi_1.0/usestd_False/exp_val_False/anneal_True_T_10/trials500_sims5000/",\
			 "NoHebb_old": "../opal/sims/results/simplemod/Bayes-SA/rmag_1_lmag_0_mag_1/phi_1.0/usestd_False/exp_val_False/anneal_True_T_10/trials500_sims5000/",\
			 "alu": "../bogacz/results/complexity_500_5000/",
			 "alu-1": "../bogacz/results/complexity_500_5000/",
			 "Bmod": "../opal/sims/results/simplemod/Bayes-SA/rmag_1_lmag_0_mag_1/phi_1.0/usestd_False/exp_val_False/anneal_True_T_10/trials500_sims5000/",
			 "SACrit_BayesAnneal": "../opal/sims/results/revisions/SA/rmag_1_lmag_0_mag_1/phi_1.0/anneal_True_T_10/use_var_True/decay_to_prior_False_gamma_10/pgrad_False/trials1000_sims1000/",
			 "SACrit_StdAnneal": "../opal/sims/results/revisions/SA/rmag_1_lmag_0_mag_1/phi_1.0/anneal_True_T_10/use_var_False/decay_to_prior_False_gamma_10/pgrad_False/trials1000_sims1000/",
			 "BayesCrit_BayesAnneal": "../opal/sims/results/revisions/Bayes-SA/rmag_1_lmag_0_mag_1/phi_1.0/anneal_True_T_10/use_var_True/decay_to_prior_False_gamma_10/pgrad_False/trials1000_sims1000/",
			 "BayesCrit_StdAnneal": "../opal/sims/results/revisions/Bayes-SA/rmag_1_lmag_0_mag_1/phi_1.0/anneal_True_T_10/use_var_False/decay_to_prior_False_gamma_10/pgrad_False/trials1000_sims1000/",
			 "SACrit_BayesAnneal_mod": "../opal/sims/results/revisions/SA/rmag_1_lmag_0_mag_1/phi_1.0/anneal_True_T_10/use_var_True/decay_to_prior_False_gamma_10/pgrad_False/trials1000_sims1000/",
			 "NoHebb": "../opal/sims/results/revisions/SA/rmag_1_lmag_0_mag_1/phi_1.0/anneal_True_T_10/use_var_True/decay_to_prior_False_gamma_10/pgrad_False/trials1000_sims1000/",
			 "Bmod": "../opal/sims/results/revisions/SA/rmag_1_lmag_0_mag_1/phi_1.0/anneal_True_T_10/use_var_True/decay_to_prior_False_gamma_10/pgrad_False/trials1000_sims1000/",
			 "SACrit_BayesAnneal_mod_phi0.5": "../opal/sims/results/revisions/SA/rmag_1_lmag_0_mag_1/phi_0.5/anneal_True_T_100/use_var_True/decay_to_prior_False_gamma_10/pgrad_False/trials1000_sims1000/",
			 "SACrit_BayesAnneal_mod_phi1.5": "../opal/sims/results/revisions/SA/rmag_1_lmag_0_mag_1/phi_1.5/anneal_True_T_10/use_var_True/decay_to_prior_False_gamma_10/pgrad_False/trials1000_sims1000/",
			 "RL": "../opal/sims/results/revisions/RL/trials1000_sims1000/",
			 "UCB": "../opal/sims/results/revisions/UCB/trials1000_sims1000/",
			 "UCB_opt": "../opal/sims/results/revisions/UCB/"}

	# rl + opal had same grid
	param_revisions_Bayes = params.get_params_all("bayesCrit")
	param_revisions_rl = params.get_params_all("rlCrit")
	param_opal = params.get_params_all("simplemod")
	param_alu = params.get_params_all("ALU2")
	param_alu_old = params.get_params_all("ALUold")
	params_rl =  params.get_params_all("RL")
	param_dict = {"alu": param_alu,"alu-1": param_alu_old,
		"OpAL*": param_opal,
		"OpAL": param_opal,
		"NoHebb_old": param_opal, 
		"Bmod": param_opal,
		"SACrit_BayesAnneal": param_revisions_rl,
		"SACrit_BayesAnneal_mod": param_revisions_rl,
		"NoHebb": param_revisions_rl,
		"Bmod": param_revisions_rl,
		"SACrit_BayesAnneal_mod_phi0.5": param_revisions_rl,
		"SACrit_BayesAnneal_mod_phi1.5": param_revisions_rl,
		"SACrit_StdAnneal": param_revisions_rl,
		"BayesCrit_BayesAnneal": param_revisions_Bayes,
		"BayesCrit_StdAnneal": param_revisions_Bayes,
		"RL": params_rl,
		"UCB": np.arange(1,200)/100,
		"UCB_opt": [None]}

	#create directory for complex graphs when non-existing
	save_path = "trials%d_%d/k%s/" %(start_plt,end_plt,int(k))
	os.makedirs(save_path, exist_ok=True)

	for level in np.arange(n_levels):
		
		#just run highest and lowest complexity
		level = level*4

		all_max_params = {}

		for e_idx,env in enumerate(envs):
			full_env = "%s_%d_%d" %(env,10,level+2)
			all_models = {}
			all_models_path = {}
			all_models_params = {}

			print("Full Env: " + full_env)
			sys.stdout.flush()

			for m_idx,model in enumerate(models):

				# save the AUcs for all params
				auc_learn_all = []
				param_path_all = []
				param_all = []

				for par in param_dict[model]:

					# get paths that we need
					if model == "UCB_opt":
						full_env = "%s_%d_%d" %(env,10,level+2)
						path_final = "%strials%d_sims1000/%s/auc_UCB.pkle" %(paths["UCB_opt"],end_plt,full_env)
						strpar = "UCB_opt"
					else: 
						# get params formatted correctly
						if model == "alu" or model == "alu-1":
							par_subset = np.round(par,5)
						elif model == "SACrit_StdAnneal" or model == "SACrit_BayesAnneal" or model == "SACrit_BayesAnneal_mod" or model == "SACrit_BayesAnneal_mod_phi0.5" or  model == "NoHebb" or model == "Bmod":
							par_subset = par[0:3]
						elif model == "RL":
							par_subset = par
						elif model == "UCB":
							par_subset = par
						else:
							par_subset = par[1:3]

						# # only run betas in original range at first
						# if par_subset[-1] > 5.0:
						# 	continue
						strpar = str(par_subset)

						path_final = \
							get_full_path(model,env,level,k,strpar,paths)
						# print("path for model %s: %s" %(model,path_final))

					# get modulated data 
					try:
						_, _, learn_curve, reward_curve, rnd_seed = \
						pickle.load(open(path_final,"rb"))
					except: #ALU didn't save random seed to file oops
						try:
							_, _, learn_curve, reward_curve = \
							pickle.load(open(path_final,"rb"))
						except:
							print("MISSING PARAMS")
							continue
					auc_learn, _ = getAUC(learn_curve,\
						reward_curve,start_plt,end_plt)
					print("param: " + strpar)
					print("path: " + path_final)
					print("AUC: " + str(auc_learn))

					# save that data
					auc_learn_all.append(auc_learn)
					param_path_all.append(path_final) #param for AUC
					param_all.append(par_subset)
			
				all_models[model] = auc_learn_all
				all_models_path[model] = param_path_all
				all_models_params[model] = param_all

			# plot hist for env
			max_params = hist_AUC(all_models,save_path,env,level,all_models_path,start_plt,end_plt,all_models_params,normalize=False)

			# save env params
			all_max_params[full_env] = max_params
		
		# plot corresponding parameters for each env
		swap_max_params(all_max_params,models,paths,level,k,start_plt,end_plt,save_path)


if __name__ == '__main__':
    main(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))