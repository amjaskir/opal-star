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

import enum
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import scipy.stats as stats
import pylab
from sklearn import metrics
import time

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

	path = paths[model]
	full_env = "%s_%d_%d/" %(env,10,level)

	# correct some path things for ALU
	if model == "alu":
		path = path  + env + "/" + full_env
	elif model == "alu-1":
		if env == "30":
			dummyenv = "lean"
		else:
			dummyenv = "rich"
		full_env = "%s_%d/" %(dummyenv,level)
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
	else:
		path_final = path + "k_%s/mod_beta/bmod/" %(k)

	return path_final + ext

def filter_by_index(models,all_models,all_models_path,all_models_params,keep_me,level):

	print("debugging filter_by_index")
	print(keep_me)

	# make new 
	# initialize dictionaries
	filt_models = {}
	filt_paths = {}
	filt_params = {}

	envs = ["80","30"]
	for _, model in enumerate(models):
		filt_models[model] = {}
		filt_paths[model] = {}
		filt_params[model] = {}

		for _,env in enumerate(envs):
			filt_models[model][env] = {}
			filt_paths[model][env] = {}
			filt_params[model][env] = {}

	str_level = str(level)
	for _, model in enumerate(models):

		this_keep = keep_me[model]
		# update rich
		filt_models[model]["80"][str_level] = [all_models[model]["80"][str_level][i] for i in this_keep]
		filt_paths[model]["80"][str_level] = [all_models_path[model]["80"][str_level][i] for i in this_keep]
		filt_params[model]["80"][str_level] = [all_models_params[model]["80"][str_level][i] for i in this_keep]

		# update lean
		filt_models[model]["30"][str_level] = [all_models[model]["30"][str_level][i] for i in this_keep]
		filt_paths[model]["30"][str_level] = [all_models_path[model]["30"][str_level][i] for i in this_keep]
		filt_params[model]["30"][str_level] = [all_models_params[model]["30"][str_level][i] for i in this_keep]
	
	return(filt_models,filt_paths,filt_params)

def filter_me(thresh,models,all_models,level,all_models_params):
	"""
		Filter out values greater than top 10% in both environments
	"""

	str_level = str(level)
	keep_me_union = {}
	keep_me_intersect = {}
	auc_order = {}
	auc_cross_values_unsorted = {}
	rich_selected_params = {}
	lean_selected_params = {}
	for _, model in enumerate(models):

		# Get top thesh params for RICH -------------------------------------
		# sort AUCs for rich
		get_me_rich = all_models[model]["80"][str_level]
		rich_sort_idx = np.flip(np.argsort(get_me_rich))	# AUC order highest to lowest

		# calc how many to select
		n_items = len(get_me_rich)
		n_select = round(n_items*thresh)

		# select the highest AUC according to position in list
		rich_selected = rich_sort_idx[0:n_select]
		# rich_selected_params[model] = [all_models_params[model]["80"][i] for i in rich_selected]


		# Get top thesh params for LEAN -------------------------------------
		# sort AUCs for rich
		get_me_lean = all_models[model]["30"][str_level]
		lean_sort_idx = np.flip(np.argsort(get_me_lean))	# AUC order highest to lowest

		# calc how many to select
		n_items = len(get_me_lean)
		n_select = round(n_items*thresh)

		# select the with highest AUC according to position in list
		lean_selected = lean_sort_idx[0:n_select]
		# lean_selected_params[model] = [all_models_params[model]["30"][i] for i in lean_selected]

		# Compare RICH and LEAN --------------------------------------------
		# position in list is rich and lean map to same parameter
		keep_me_union[model] = list(set(rich_selected) | set(lean_selected))
		keep_me_intersect[model] = [x for x in rich_selected if x in lean_selected]

		# Get order of cross env performance --------------------------------
		auc_cross = np.array(get_me_rich) + np.array(get_me_lean)
		auc_order[model] = np.flip(np.argsort(auc_cross)) # parameter index of sorted auc order
		auc_cross_values_unsorted[model] = auc_cross # auc NOT FILTERED!!!!

	return (keep_me_union,keep_me_intersect,auc_order,auc_cross_values_unsorted)


def hist_AUC(models,save_path,env,level,legend_dict,normalize=False):
	""" 
		Histogram of AUC under reward curve
		example sim1 - mod true
				sim2 - mod false
		what - which comparison is being done
			 - see function get_settings()
		normalize - normalize by number of parameters
	"""

	envs = ["80","30"]

	fnt_sz = 20
	for env in envs:
		# set up fig
		fig, ax = plt.subplots(figsize=(6, 5))
		for key in models:
			# plot
			histme = models[key][env][str(level)]
			ax.hist(histme,alpha=0.5,density=normalize,label=legend_dict[key])

		ax.set_ylabel("frequency",fontsize = fnt_sz)
		ax.set_xlabel("AUC",fontsize = fnt_sz)

		if env == "30":
			plt.title("Lean Environment, %s options" %(level))
		else:
			plt.title("Rich Environment, %s options" %(level))
		plt.legend(fontsize=14)

		# save
		plt.savefig(save_path + "filtered_leaning_%s_%d" %(env,level), dpi = 400, bbox_inches='tight')
		plt.close()

def plot_opt(models,level,all_models,all_models_path,all_models_params,end_plt,save_path,legend_dict):
	'''
	get parameters which optimize the AUC across both environments
	for each model. Saves learning curve
	'''

	plt.rcParams.update({'font.size': 18}) # font size 
	str_level = str(level)

	# rich curve
	fig_rich_total, ax_rich_total = plt.subplots(figsize=(6, 5)) #total = cross env
	fig_rich, ax_rich = plt.subplots(figsize=(6, 5))

	# lean curve
	fig_lean_total, ax_lean_total = plt.subplots(figsize=(6, 5))
	fig_lean, ax_lean = plt.subplots(figsize=(6, 5))

	# save params and AUC values which opt for a specific env or across env
	# for FIXED complexity and FIXED time horizon
	max_param_dict = {}
	for _,model in enumerate(models):
		max_param_dict[model] = {}
		max_param_dict[model]["rich"] = {}
		max_param_dict[model]["lean"] = {}
		max_param_dict[model]["cross"] = {}

	for _, model in enumerate(models):

		# get the params with best AUC across rich and lean
		rich_AUCs = all_models[model]["80"][str_level]
		rich_paths = all_models_path[model]["80"][str_level]
		rich_params = all_models_params[model]["80"][str_level]	# rich and lean parameters are the same

		lean_AUCs = all_models[model]["30"][str_level]
		lean_paths = all_models_path[model]["30"][str_level]
		lean_params = all_models_params[model]["30"][str_level]

		total_AUCs = np.array(rich_AUCs) + np.array(lean_AUCs)	# cross env AUC

		max_param = np.where(np.max(total_AUCs) == total_AUCs)[0][0] #TODO: print if multiple
		max_param_rich = np.where(np.max(rich_AUCs) == rich_AUCs)[0][0] #TODO: print if multiple
		max_param_lean = np.where(np.max(lean_AUCs) == lean_AUCs)[0][0] #TODO: print if multiple
		opt_param_total = str(rich_params[max_param])	# same as if indexing lean, sanity check
		opt_param_rich = str(rich_params[max_param_rich])
		opt_param_lean = str(lean_params[max_param_lean])	

		# print info so I can visually compare AUCs
		print("Model: " + model)
		print("Max param total: " + opt_param_total)
		print("Max param rich: " + opt_param_rich)
		print("Max param lean: " + opt_param_lean) 
		print("Max AUC: " + str(np.max(total_AUCs)))

		# save for posterity ------------------------------------------------------------------
		sort_rich_idx = np.flip(np.argsort(rich_AUCs)) # order from largest to smallest
		sort_lean_idx = np.flip(np.argsort(lean_AUCs))
		sort_total_idx = np.flip(np.argsort(total_AUCs))

		max_param_dict[model]["rich"]["AUCs"] = [rich_AUCs[i] for i in sort_rich_idx]
		max_param_dict[model]["rich"]["params"] = [rich_params[i] for i in sort_rich_idx]
		max_param_dict[model]["lean"]["AUCs"] = [lean_AUCs[i] for i in sort_lean_idx]
		max_param_dict[model]["lean"]["params"] = [lean_params[i] for i in sort_lean_idx]
		max_param_dict[model]["cross"]["AUCs"] = [total_AUCs[i] for i in sort_total_idx]
		max_param_dict[model]["cross"]["params"] = [rich_params[i] for i in sort_total_idx] #rich and lean are the same
		# -------------------------------------------------------------------------------------


		# get the appropriate curves for cross-env/total --------------------------------------
		opt_rich_path = rich_paths[max_param]
		opt_lean_path = lean_paths[max_param]

		_, _, rich_curve, _, _ = \
						pickle.load(open(opt_rich_path,"rb"))
		_, _, lean_curve, _, _ = \
						pickle.load(open(opt_lean_path,"rb"))

		(avg_sm_rich,sem_sm_rich) = rich_curve
		(avg_sm_lean,sem_sm_lean) = lean_curve

		# index to the horizon we want
		(avg_sm_rich,sem_sm_rich) = (avg_sm_rich[0:end_plt],sem_sm_rich[0:end_plt])
		(avg_sm_lean,sem_sm_lean) = (avg_sm_lean[0:end_plt],sem_sm_lean[0:end_plt])

		# plot!
		xaxis = np.arange(len(avg_sm_rich))
		ax_rich_total.errorbar(xaxis,avg_sm_rich,yerr=sem_sm_rich,label=legend_dict[model],linewidth=2.)
		ax_lean_total.errorbar(xaxis,avg_sm_lean,yerr=sem_sm_lean,label=legend_dict[model],linewidth=2.)

		# get for env-only ---------------------------------------------------------------
		opt_rich_path = rich_paths[max_param_rich]
		opt_lean_path = lean_paths[max_param_lean]

		_, _, rich_curve, _, _ = \
						pickle.load(open(opt_rich_path,"rb"))
		_, _, lean_curve, _, _ = \
						pickle.load(open(opt_lean_path,"rb"))

		(avg_sm_rich,sem_sm_rich) = rich_curve
		(avg_sm_lean,sem_sm_lean) = lean_curve

		# index to the horizon we want
		(avg_sm_rich,sem_sm_rich) = (avg_sm_rich[0:end_plt],sem_sm_rich[0:end_plt])
		(avg_sm_lean,sem_sm_lean) = (avg_sm_lean[0:end_plt],sem_sm_lean[0:end_plt])

		# plot!
		xaxis = np.arange(len(avg_sm_rich))
		ax_rich.errorbar(xaxis,avg_sm_rich,yerr=sem_sm_rich,label=legend_dict[model],linewidth=2.)
		ax_lean.errorbar(xaxis,avg_sm_lean,yerr=sem_sm_lean,label=legend_dict[model],linewidth=2.)

	# aesthetics - rich
	ax_rich_total.set_ylabel("p(best)")
	ax_rich_total.set_xlabel("trial")
	# ax_rich.set_title("Rich Environment, %s options" %(level))
	ax_rich_total.legend(loc='lower right')

	ax_rich.set_ylabel("p(best)")
	ax_rich.set_xlabel("trial")
	# ax_rich.set_title("Rich Environment, %s options" %(level))
	ax_rich.legend(loc='lower right')

	# aesthetics - lean
	ax_lean_total.set_ylabel("p(best)")
	ax_lean_total.set_xlabel("trial")
	# ax_lean.set_title("Lean Environment, %s options" %(level))
	# ax_lean.legend(loc='lower right')

	ax_lean.set_ylabel("p(best)")
	ax_lean.set_xlabel("trial")

	# save
	fig_rich_total.savefig(save_path + "optcurves_rich_%d" %(level), dpi = 400)
	plt.close(fig_rich_total)

	fig_lean_total.savefig(save_path + "optcurves_lean_%d" %(level), dpi = 400)
	plt.close(fig_lean_total)

	fig_rich.savefig(save_path + "optcurves_rich_envONLY%d" %(level), dpi = 400)
	plt.close(fig_rich_total)

	fig_lean.savefig(save_path + "optcurves_lean_envONLY%d" %(level), dpi = 400)
	plt.close(fig_lean)

	# save max params 
	prepare_to_dump =  save_path + "max_param_complexity%d.pkle" %(level)
	pickle.dump(max_param_dict, open(prepare_to_dump,"wb"))


def plot_opt_total(models,auc_cross_values_unsorted_by_level,all_models_path,all_models_params,end_plt,save_path,legend_dict,n_levels):
	'''
	get parameters which optimize the AUC across both environments AND level
	for each model. Saves learning curve
	'''

	plt.rcParams.update({'font.size': 18}) # font size 

	# set up subplots ------------------------------------------------------------------
	# rich curve
	fig_rich = plt.figure(figsize = (6,5*n_levels))
	gs = fig_rich.add_gridspec(n_levels,1)
	ax_rich = []
	for level in np.arange(n_levels):
		ax_rich.append(fig_rich.add_subplot(gs[level, 0]))

		# aesthetics for level-wise
		ax_rich[level].set_ylabel("p(best)")
		ax_rich[level].set_xlabel("trial")
		ax_rich[level].set_title("Rich, %s options" %((2 if level == 0 else 6))) #level+2
	plt.subplots_adjust(hspace=0.4)

	# lean curve
	fig_lean = plt.figure(figsize = (6,5*n_levels))
	gs = fig_lean.add_gridspec(n_levels,1)
	ax_lean = []
	for level in np.arange(n_levels):
		ax_lean.append(fig_lean.add_subplot(gs[level, 0]))

		# aesthetics
		ax_lean[level].set_ylabel("p(best)")
		ax_lean[level].set_xlabel("trial")
		ax_lean[level].set_title("Lean, %s options" %((2 if level == 0 else 6))) #level+2
	plt.subplots_adjust(hspace=0.4)
	# ------------------------------------------------------------------------------------
		

	# calculate the things
	all_max_params = {}
	all_rich_plus_lean = {}
	opt_curves_rich = {}
	opt_curves_lean = {}
	for _, model in enumerate(models):

		opt_curves_rich[model] = np.zeros([n_levels, end_plt])
		opt_curves_lean[model] = np.zeros([n_levels, end_plt])
		rich_plus_lean = []
		for level in np.arange(n_levels):
			next_level = (2 if level == 0 else 6) #level+2
			str_level = str(next_level)

			# get the params with best AUC across rich and lean
			if level ==  0:
				rich_plus_lean = auc_cross_values_unsorted_by_level[str_level][model]
			else:
				rich_plus_lean = rich_plus_lean + auc_cross_values_unsorted_by_level[str_level][model] # add across levels
		
		all_rich_plus_lean[model] = rich_plus_lean # save for later
		max_param = np.where(np.max(rich_plus_lean) == rich_plus_lean)[0][0] #TODO: print if multiple
		opt_param_rich = str(all_models_params[model]["80"]["2"][max_param])
		opt_param_lean = str(all_models_params[model]["30"]["6"][max_param]) # should be sane independent of level or env, sanity check
		all_max_params[model] = all_models_params[model]["80"]["2"][max_param] # save max params

		# print info so I can visually compare AUCs
		print("Model: " + model)
		print("Max param: " + opt_param_rich) # this should be the same as below
		print("Max param: " + opt_param_lean) 
		print("Max AUC: " + str(np.max(rich_plus_lean)))


		# plot what this opt param looks like on a level by level
		for level in np.arange(n_levels):
			next_level = (2 if level == 0 else 6) #level+2
			str_level = str(next_level)

			# get paths for level for plotting	
			rich_paths = all_models_path[model]["80"][str_level]
			lean_paths = all_models_path[model]["30"][str_level]

			# get the appropriate curves
			opt_rich_path = rich_paths[max_param]
			opt_lean_path = lean_paths[max_param]

			_, _, rich_curve, _, _ = \
							pickle.load(open(opt_rich_path,"rb"))
			_, _, lean_curve, _, _ = \
							pickle.load(open(opt_lean_path,"rb"))

			(avg_sm_rich,sem_sm_rich) = rich_curve
			(avg_sm_lean,sem_sm_lean) = lean_curve

			# index to the horizon we want
			(avg_sm_rich,sem_sm_rich) = (avg_sm_rich[0:end_plt],sem_sm_rich[0:end_plt])
			(avg_sm_lean,sem_sm_lean) = (avg_sm_lean[0:end_plt],sem_sm_lean[0:end_plt])

			# plot!
			xaxis = np.arange(len(avg_sm_rich))
			ax_rich[level].errorbar(xaxis,avg_sm_rich,yerr=sem_sm_rich,label=legend_dict[model],linewidth=2.)
			ax_lean[level].errorbar(xaxis,avg_sm_lean,yerr=sem_sm_lean,label=legend_dict[model],linewidth=2.)

			# save for avg curve
			opt_curves_rich[model][level,:]  = avg_sm_rich
			opt_curves_lean[model][level,:]  = avg_sm_lean


	# make sure legends show up
	for level in np.arange(n_levels):
		ax_rich[level].legend(loc='lower right') 
		ax_lean[level].legend(loc='lower right')
	
	# save
	fig_rich.savefig(save_path + "opt_total_curves_rich", dpi = 400)
	plt.close(fig_rich)

	fig_lean.savefig(save_path + "opt_total_curves_lean", dpi = 400)
	plt.close(fig_lean)


	# aesthetics for averaged opt total curves
	fig_all,ax_all = plt.subplots(figsize = (6,5))
	for _, model in enumerate(models):
		all_curves = np.concatenate((opt_curves_rich[model], opt_curves_lean[model]))

		avg_opt_total_curves = np.mean(all_curves, axis = 0)
		sem_opt_total_curves = stats.sem(all_curves, axis = 0)

		# plot avg
		_, caps, bars = ax_all.errorbar(xaxis,avg_opt_total_curves,yerr=sem_opt_total_curves,label=legend_dict[model],linewidth=2.)

		# make error bars translucent
		[bar.set_alpha(0.5) for bar in bars]
		[cap.set_alpha(0.5) for cap in caps]

	# save me
	ax_all.legend(loc='lower right')
	fig_all.savefig(save_path + "opt_avg_total", dpi = 400)
	plt.close(fig_all)

	# plot without SEM just to see what it looks like
	# aesthetics for averaged opt total curves
	fig_all_noSEM,ax_all_noSEM = plt.subplots(figsize = (6,5))
	for _, model in enumerate(models):
		all_curves = np.concatenate((opt_curves_rich[model], opt_curves_lean[model]))

		avg_opt_total_curves = np.mean(all_curves, axis = 0)
		sem_opt_total_curves = stats.sem(all_curves, axis = 0)

		# plot avg
		ax_all_noSEM.errorbar(xaxis,avg_opt_total_curves,label=legend_dict[model],linewidth=2.)

	# save me
	ax_all_noSEM.legend(loc='lower right')
	fig_all_noSEM.savefig(save_path + "opt_avg_total_noSEM", dpi = 400)
	plt.close(fig_all_noSEM)

	# save max params for reproducibility
	prepare_to_dump =  save_path + "max_params.pkle"
	pickle.dump(all_max_params, open(prepare_to_dump,"wb"))

	# save auc values for reproducibility, used in plot_opt_conditional
	prepare_to_dump =  save_path + "rich_plus_lean_by_model.pkle"
	pickle.dump(all_rich_plus_lean, open(prepare_to_dump,"wb"))


def plot_opt_total_conditional(models,all_models_path,all_models_params,end_plt,save_path,search_path,legend_dict,n_levels):
	'''
	load parameters which optimize the AUC across both environments AND level
	for each model, saved from other calls of plot_opt_total calls.

	then, get parameters which optimize across ENV X LEVEL X TIME

	For time, we consider 250 trials and 1000 trials ONLY.

	This function will SLEEP until desired files become available to handle
	asynchronous processing by levels
	'''
	plt.rcParams.update({'font.size': 18}) # font size 

	time_horizons = [250, 1000]

	# set up subplots ------------------------------------------------------------------
	# rich curve
	fig_rich = plt.figure(figsize = (6,5*n_levels))
	gs = fig_rich.add_gridspec(n_levels,1)
	ax_rich = []
	for level in np.arange(n_levels):
		ax_rich.append(fig_rich.add_subplot(gs[level, 0]))

		# aesthetics for level-wise
		ax_rich[level].set_ylabel("p(best)")
		ax_rich[level].set_xlabel("trial")
		ax_rich[level].set_title("Rich, %s options" %((2 if level == 0 else 6))) #level+2
	plt.subplots_adjust(hspace=0.4)

	# lean curve
	fig_lean = plt.figure(figsize = (6,5*n_levels))
	gs = fig_lean.add_gridspec(n_levels,1)
	ax_lean = []
	for level in np.arange(n_levels):
		ax_lean.append(fig_lean.add_subplot(gs[level, 0]))

		# aesthetics
		ax_lean[level].set_ylabel("p(best)")
		ax_lean[level].set_xlabel("trial")
		ax_lean[level].set_title("Lean, %s options" %((2 if level == 0 else 6))) #level+2
	plt.subplots_adjust(hspace=0.4)
	# ------------------------------------------------------------------------------------
		

	# calculate the things
	all_max_params = {}
	all_rich_plus_lean = {}
	opt_curves_rich = {}
	opt_curves_lean = {}
	for _, model in enumerate(models):

		opt_curves_rich[model] = np.zeros([n_levels, end_plt])
		opt_curves_lean[model] = np.zeros([n_levels, end_plt])
		rich_plus_lean = []
		for h_idx, horizon in enumerate(time_horizons):

			load_me = search_path + "%d/rich_plus_lean_by_model.pkle" %(horizon)
			attempt_num = 0
			success = False
			while not success:
				try:
					rich_plus_lean_load = pickle.load(open(load_me,"rb"))
					success = True
				except:
					time.sleep(60) # wait a minute
					attempt_num += 1
				
				# too many attempts? throw an error
				if attempt_num > 5:
					err = 'Waited and cannot find file: %s' %(load_me)
					raise Exception(err)

			# get the params with best AUC across rich and lean
			if h_idx ==  0:
				rich_plus_lean = rich_plus_lean_load[model]
			else:
				rich_plus_lean = rich_plus_lean + rich_plus_lean_load[model] # add across levels
		
		all_rich_plus_lean[model] = rich_plus_lean # save for later
		max_param = np.where(np.max(rich_plus_lean) == rich_plus_lean)[0][0] #TODO: print if multiple
		opt_param_rich = str(all_models_params[model]["80"]["2"][max_param])
		opt_param_lean = str(all_models_params[model]["30"]["6"][max_param]) # should be sane independent of level or env, sanity check
		all_max_params[model] = all_models_params[model]["80"]["2"][max_param] # save max params

		# print info so I can visually compare AUCs
		print("Model: " + model)
		print("Max param ENV LEVEL TIME: " + opt_param_rich)
		print("Max param ENV LEVEL TIME: " + opt_param_lean) 
		print("Max AUC: " + str(np.max(rich_plus_lean)))


		# plot what this opt param looks like on a level by level
		for level in np.arange(n_levels):
			next_level = (2 if level == 0 else 6) #level+2
			str_level = str(next_level)

			# get paths for level for plotting	
			rich_paths = all_models_path[model]["80"][str_level]
			lean_paths = all_models_path[model]["30"][str_level]

			# get the appropriate curves
			opt_rich_path = rich_paths[max_param]
			opt_lean_path = lean_paths[max_param]

			_, _, rich_curve, _, _ = \
							pickle.load(open(opt_rich_path,"rb"))
			_, _, lean_curve, _, _ = \
							pickle.load(open(opt_lean_path,"rb"))

			(avg_sm_rich,sem_sm_rich) = rich_curve
			(avg_sm_lean,sem_sm_lean) = lean_curve

			# index to the horizon we want
			(avg_sm_rich,sem_sm_rich) = (avg_sm_rich[0:end_plt],sem_sm_rich[0:end_plt])
			(avg_sm_lean,sem_sm_lean) = (avg_sm_lean[0:end_plt],sem_sm_lean[0:end_plt])

			# plot!
			xaxis = np.arange(len(avg_sm_rich))
			ax_rich[level].errorbar(xaxis,avg_sm_rich,yerr=sem_sm_rich,label=legend_dict[model],linewidth=2.)
			ax_lean[level].errorbar(xaxis,avg_sm_lean,yerr=sem_sm_lean,label=legend_dict[model],linewidth=2.)

			# save for avg curve
			opt_curves_rich[model][level,:]  = avg_sm_rich
			opt_curves_lean[model][level,:]  = avg_sm_lean


	# make sure legends show up
	for level in np.arange(n_levels):
		ax_rich[level].legend(loc='lower right') 
		ax_lean[level].legend(loc='lower right')
	
	# save
	fig_rich.savefig(save_path + "opt_total_ELT_curves_rich", dpi = 400)
	plt.close(fig_rich)

	fig_lean.savefig(save_path + "opt_total_ELT_curves_lean", dpi = 400)
	plt.close(fig_lean)


	# aesthetics for averaged opt total curves
	fig_all,ax_all = plt.subplots(figsize = (6,5))
	for _, model in enumerate(models):
		all_curves = np.concatenate((opt_curves_rich[model], opt_curves_lean[model]))

		avg_opt_total_curves = np.mean(all_curves, axis = 0)
		sem_opt_total_curves = stats.sem(all_curves, axis = 0)

		# plot avg
		_, caps, bars = ax_all.errorbar(xaxis,avg_opt_total_curves,yerr=sem_opt_total_curves,label=legend_dict[model],linewidth=2.)

		# make error bars translucent
		[bar.set_alpha(0.5) for bar in bars]
		[cap.set_alpha(0.5) for cap in caps]

	# save me
	ax_all.legend(loc='lower right')
	fig_all.savefig(save_path + "opt_avg_total_ELT", dpi = 400)
	plt.close(fig_all)

	# plot without SEM just to see what it looks like
	# aesthetics for averaged opt total curves
	fig_all_noSEM,ax_all_noSEM = plt.subplots(figsize = (6,5))
	for _, model in enumerate(models):
		all_curves = np.concatenate((opt_curves_rich[model], opt_curves_lean[model]))

		avg_opt_total_curves = np.mean(all_curves, axis = 0)
		sem_opt_total_curves = stats.sem(all_curves, axis = 0)

		# plot avg
		ax_all_noSEM.errorbar(xaxis,avg_opt_total_curves,label=legend_dict[model],linewidth=2.)

	# save me
	ax_all_noSEM.legend(loc='lower right')
	fig_all_noSEM.savefig(save_path + "opt_avg_total_noSEM_ELT", dpi = 400)
	plt.close(fig_all_noSEM)

	# save max params for reproducibility
	prepare_to_dump =  save_path + "max_params_ELT.pkle"
	pickle.dump(all_max_params, open(prepare_to_dump,"wb"))

def main(thresh,start_plt,end_plt):
	
	k = 20
	n_levels = 2 #5 # number of levels of complexity # 2 - just run low and high

	# model paths
	envs = ["80","30"]
	# models = ["opal*","Balanced","NoHebb","alu","alu-1"]
	# models = ["opal*","Balanced","NoHebb","Bmod"]
	# models = ["OpAL*","OpAL","NoHebb"]
	# models = ["UCB", "RL","OpAL*", "OpAL"]
	# models = ["UCB", "RL","NoHebb","SACrit_BayesAnneal", "SACrit_BayesAnneal_mod"]
	# models = ["RL","SACrit_BayesAnneal_mod"]
	# models = ["UCB","SACrit_BayesAnneal","SACrit_BayesAnneal_mod","RL","NoHebb","Bmod","alu"]
	models = ["UCB","RL","SACrit_BayesAnneal_mod"]
	# models = ["SACrit_BayesAnneal_mod","UCB"]
	model_str = "revisions2/paper/"
	# models = ["OpAL", "BayesCrit_StdAnneal"]
	paths = {"OpAL*": "../opal/sims/results/simplemod/Bayes-SA/rmag_1_lmag_0_mag_1/phi_1.0/usestd_False/exp_val_False/anneal_True_T_10/trials500_sims5000/",\
			 "OpAL": "../opal/sims/results/simplemod/Bayes-SA/rmag_1_lmag_0_mag_1/phi_1.0/usestd_False/exp_val_False/anneal_True_T_10/trials500_sims5000/",\
			 "NoHebb_old": "../opal/sims/results/simplemod/Bayes-SA/rmag_1_lmag_0_mag_1/phi_1.0/usestd_False/exp_val_False/anneal_True_T_10/trials500_sims5000/",\
			 "alu": "../bogacz/results/complexity_500_5000/",
			 "alu-1": "../bogacz/results/complexity_500_5000/",
			 "Bmod_old": "../opal/sims/results/simplemod/Bayes-SA/rmag_1_lmag_0_mag_1/phi_1.0/usestd_False/exp_val_False/anneal_True_T_10/trials500_sims5000/",
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
			 "UCB_old": "../opal/sims/results/revisions/UCB/trials1000_sims1000/",
			 "UCB": "../opal/sims/results/revisions2/UCB/trials1000_sims1000/",
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
		"Bmod_old": param_opal,
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

	# for plotting purposes
	legend_dict = {
		"Bmod": "Beta Modulation",
		"SACrit_BayesAnneal": "OpAL+",
		"SACrit_BayesAnneal_mod": "OpAL*",
		"NoHebb": "No Hebb",
		"RL": "Q-learner",
		"UCB": "UCB"
		}

	#create directory for complex graphs when non-existing
	save_path = "filter_%.2f/%s/trials%d_%d/" %(thresh,model_str,start_plt,end_plt)
	search_path = "filter_%.2f/%s/trials%d_" %(thresh,model_str,start_plt)
	os.makedirs(save_path, exist_ok=True)


	# initialize dictionaries
	all_models = {}
	all_models_path = {}
	all_models_params = {}

	for _, model in enumerate(models):
		all_models[model] = {}
		all_models_path[model] = {}
		all_models_params[model] = {}

		for _,env in enumerate(envs):
			all_models[model][env] = {}
			all_models_path[model][env] = {}
			all_models_params[model][env] = {}
	
	keep_me_union_by_level = {}
	keep_me_auc_order_by_level = {}
	auc_cross_values_unsorted_by_level = {} #NOT FILTERED!!!!

	for level in np.arange(n_levels):
		level = (2 if level == 0 else 6) #level+2 
		str_level = str(level)

		for e_idx,env in enumerate(envs):
			print("environment: %s"%env)
			full_env = "%s_%d_%d" %(env,10,level)

			for m_idx,model in enumerate(models):

				# save the AUcs for all params
				auc_learn_all = []
				param_path_all = []
				param_all = []

				for par in param_dict[model]:

					# get paths that we need
					if model == "UCB_opt":
						full_env = "%s_%d_%d" %(env,10,level)
						path_final = "%strials%d_sims1000/%s/auc_UCB.pkle" %(paths["UCB_opt"],end_plt,full_env)
						strpar = "UCB_opt"
					else: 
						# get params formatted correctly
						if model == "alu" or model == "alu-1":
							par_subset = np.round(par,5)
						elif model == "SACrit_StdAnneal" or model == "SACrit_BayesAnneal" or model == "SACrit_BayesAnneal_mod" or model == "SACrit_BayesAnneal_mod_phi0.5" or model == "SACrit_BayesAnneal_mod_phi1.5" or model == "NoHebb" or model == "Bmod":
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
						_, _, learn_curve, reward_curve, _ = \
						pickle.load(open(path_final,"rb"))
					except: #ALU didn't save random seed to file oops
						try:
							_, _, learn_curve, reward_curve = \
							pickle.load(open(path_final,"rb"))
						except:
							print("MISSING PARAMS")
							print(model)
							print(path_final)
							continue
					auc_learn, _ = getAUC(learn_curve,\
						reward_curve,start_plt,end_plt)

					# save that data
					auc_learn_all.append(auc_learn)
					param_path_all.append(path_final) #param for AUC
					param_all.append(par_subset)
			
				all_models[model][env][str_level] = auc_learn_all
				all_models_path[model][env][str_level] = param_path_all
				all_models_params[model][env][str_level] = param_all

		
		# get parameter index to filter
		keep_me_union,keep_me_intersect,auc_order,auc_cross_values_unsorted = filter_me(thresh,models,all_models,level,all_models_params)
		keep_me_union_by_level[str(level)] = keep_me_union
		keep_me_auc_order_by_level[str(level)] = auc_order
		auc_cross_values_unsorted_by_level[str(level)] = auc_cross_values_unsorted # NOT FILTERED!!!
		plot_opt(models,level,all_models,all_models_path,all_models_params,end_plt,save_path,legend_dict)

		# filter and plot hist for env
		filt_models,_,filt_params = filter_by_index(models,all_models,all_models_path,all_models_params,keep_me_intersect,level)
		hist_AUC(filt_models,save_path + "intersect_",env,level,legend_dict,normalize=True)

		# save text of overlapping params to easily visualize
		overlapParams =  save_path + "intersect_complexity%d.txt" %(level)
		with open(overlapParams, "w") as text_file:
			for _, model in enumerate(models):
				text_file.write(model + "\n")
				text_file.write(str(filt_params[model]))
				text_file.write("\n")
		# also pickle for ease of accessing later
		overlapParams =  save_path + "intersect_complexity%d.pkle" %(level)
		pickle.dump(filt_params, open(overlapParams,"wb"))


	# now filter by union across complexities
	keep_me_union_all_complexities = {}
	for _, model in enumerate(models):
		if n_levels == 2:
			keep_me_union_all_complexities[model] = list(set(keep_me_union_by_level["2"][model]) | set(keep_me_union_by_level["6"][model]))
		else:
			keep_me_union_all_complexities[model] = list(set(keep_me_union_by_level["2"][model]) | set(keep_me_union_by_level["3"][model]) | set(keep_me_union_by_level["4"][model]) | set(keep_me_union_by_level["5"][model]) | set(keep_me_union_by_level["6"][model]))

	ordered_params = {}
	for level in np.arange(n_levels):
		level = (2 if level == 0 else 6) #level+2
		# filter by union
		filt_models,_,filt_params = filter_by_index(models,all_models,all_models_path,all_models_params,keep_me_union_all_complexities,level)
		hist_AUC(filt_models,save_path + "union_",env,level,legend_dict,normalize=True)

		# union is across complexities, so only save once
		if level == 0:
			# save text of overlapping params to easily visualize
			overlapParams =  save_path + "union.txt"
			with open(overlapParams, "w") as text_file:
				for _, model in enumerate(models):
					text_file.write(model + "\n")
					text_file.write(str(filt_params[model]))
					text_file.write("\n")
			# also pickle for ease of accessing later
			overlapParams =  save_path + "union.pkle"
			pickle.dump(filt_params, open(overlapParams,"wb"))

		# filter by AUC order and save
		_,_,res_params = filter_by_index(models,all_models,all_models_path,all_models_params,keep_me_auc_order_by_level[str(level)],level)
		ordered_params[str(level)] = {}
		for _,model in enumerate(models):	#reformat
			ordered_params[str(level)][model] = res_params[model]["80"][str(level)]	#80 and 30 are the same

	# save AUC ordering
	auc_Path =  save_path + "auc_order.pkle"
	pickle.dump(ordered_params, open(auc_Path,"wb"))


	# get top parameter across env AND level
	plot_opt_total(models,auc_cross_values_unsorted_by_level,all_models_path,all_models_params,end_plt,save_path,legend_dict,n_levels)

	# plot top across env AND level AND time if available
	plot_opt_total_conditional(models,all_models_path,all_models_params,end_plt,save_path,search_path,legend_dict,n_levels)


		

if __name__ == '__main__':
    main(float(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))