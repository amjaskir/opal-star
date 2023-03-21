##################################################################
# Helper script for plotting histogram of AUC under averaged
# learning and reward curves across parameter a grid search
# of parameter settings 
#
# file: helpers/params.py
##################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import scipy.stats as stats
import os

def get_settings(what):
	""" get plot settings for specified comparison
		- mod, comparing DA modulated vs balanced models
		- norm, comparing normalized vs unnormalized models
		- env, comparing perf in rich and lean environments
	"""
	if what == "mod":
		comparison = "OpAL* - OpAL+ AUC"
		base = "OpAL+ AUC"
		c1 = (137/255, 0, 1, 0.5) #purple, DA mod
		c2 = (64/255, 64/255, 64/255, 0.5) #grey, balanced
	elif what == "balanced":
		comparison = "OpAL* - OpAL+ AUC"
		base = "OpAL+ AUC"
		c1 = (137/255, 0, 1, 0.5) #purple, DA mod
		c2 = (64/255, 64/255, 64/255, 0.5) #grey, balanced
	elif what == "nohebb":
		comparison = "OpAL* - No Hebb AUC"
		base = "No Hebb AUC"
		c1 = (137/255, 0, 1, 0.5) #purple, DA mod
		c2 = (64/255, 64/255, 64/255, 0.5) #grey, no hebb
	elif what == "norm":
		comparison = "Normalized - Unnormalized AUC"
		base = "Unnormalized AUC"
		c1 = (0, 128/255, 255/255, 0.5) #blue, normalized
		c2 = (64/255, 64/255, 64/255, 0.5) #grey, unnormalized
	elif what == "lrate":
		comparison = "DA - Lrate Mod AUC"
		base = "Lrate Mod AUC"
		c1 = (0, 128/255, 255/255, 0.5) #blue, normalized
		c2 = (64/255, 64/255, 64/255, 0.5) #grey, unnormalized
	elif what == "bmod":
		comparison = "OpAL* - Beta Mod AUC"
		base = "Beta Mod AUC"
		c1 = (0, 204/255, 0, 0.5) #green, rich env
		c2 = (64/255, 64/255, 64/255, 0.5) #grey, unnormalized
	elif what == "env":
		comparison = "Rich - Lean AUC"
		base = "Lean AUC"
		c1 = (0, 204/255, 0, 0.5) #green, rich env
		c2 = (64/255, 64/255, 64/255, 0.5) #grey, lean env
	else:
		err = 'Invalid value given for arg what. \"%s\" given' %what
		raise Exception(err)

	return (comparison,base,c1,c2)


def learn_curve(curve1,auc1,curve2,auc2,par,path,w_auc=False,ax = None,pltn=None, ylim=None, ttle = True):
	""" 
		Graph learning curve for specified parameters
		curve1 	- modulated model curve
		curve2 	- comparison model curve
		par  	- parameters
		path 	- save path
		w_auc 	- bool, include AUC in legend
		pltn 	- part of curve to plot
	"""

	(avg_sm1,sem_sm1) = curve1   # mod
	(avg_sm2,sem_sm2) = curve2   # balanced

	# restrict to pltn if specified
	if pltn is not None:
		(avg_sm1,sem_sm1) = (avg_sm1[0:pltn],sem_sm1[0:pltn])
		(avg_sm2,sem_sm2) = (avg_sm2[0:pltn],sem_sm2[0:pltn])

	# replot
	xaxis = np.arange(pltn)
	linewidth = 2.
	if ax is None:
		# plot curve and save with or without AUC totals
		plt.rcParams.update({'font.size': 22})
		fig, axs = plt.subplots(figsize=(7, 5))
		axs.errorbar(xaxis,avg_sm1,yerr=sem_sm1, c = "purple", linewidth = linewidth)
		axs.errorbar(xaxis,avg_sm2,yerr=sem_sm2, c = "black", linewidth = linewidth)
		if w_auc:
			plt.legend(("$\\rho \propto V$ AUC:%.2f" %auc1, "$\\rho=0$ AUC:%.2f" %auc2), \
				fontsize = 15, markerscale = 50, loc = 'lower right')
		# else:
			# plt.legend(("$\\rho \propto V$", "$\\rho=0$"), fontsize = 30, markerscale = 50, loc = 'lower right')
		if ttle:
			plt.title("Params:" + str(par))
		plt.xlabel("Trial")
		plt.ylabel("$\overline{p}$(best)")
		plt.tight_layout()

		if ylim is not None:
			plt.ylim(ylim)

		# save in separate curve folder
		save_me = path + "curves/"
		os.makedirs(save_me, exist_ok=True) 
		plt.savefig(save_me + str(par) + ".png")
		plt.close()
	else:
		# add plot to axis given with AUC totals
		plt.rcParams.update({'font.size': 8})
		if ylim is not None:
			plt.ylim(ylim)
		linewidth = .5
		l1 = ax.errorbar(xaxis,avg_sm1,yerr=sem_sm1, c = "purple", linewidth = linewidth)
		l2 = ax.errorbar(xaxis,avg_sm2,yerr=sem_sm2, c = "black", linewidth = linewidth)
		if ttle:
			ax.set_title(str(par))
		# l1.set_label("AUC:%.2f" %auc1)
		# l2.set_label("AUC:%.2f" %auc2)
		# ax.legend(loc = 'lower right')


def graph_learn(sim1,sim2,path,what):
	""" 
		Histogram of AUC under learning curve
		example sim1 - mod true
				sim2 - mod false
		what - which comparison is being done
			 - see function get_settings()
	"""

	comparison,base,c1,c2 = get_settings(what)

	plt.rcParams.update({'font.size': 22})
	fig, ax = plt.subplots()
	ax.hist(sim1,color = c1)
	ax.hist(sim2,color = c2)
	plt.ylabel("frequency")
	plt.xlabel("AUC")
	plt.title("Learning Curve")
	plt.tight_layout()
	plt.savefig(path + "learnhist")
	plt.close()

def graph_reward(sim1,sim2,path,what):
	""" 
		Histogram of AUC under reward curve
		example sim1 - mod true
				sim2 - mod false
		what - which comparison is being done
			 - see function get_settings()
	"""

	comparison,base,c1,c2 = get_settings(what)

	plt.rcParams.update({'font.size': 22})
	fig, ax = plt.subplots(figsize=(6, 5))
	ax.hist(sim1,color = c1)
	ax.hist(sim2,color = c2)
	plt.ylabel("frequency")
	plt.xlabel("AUC")
	plt.title("Reward Curves")
	plt.tight_layout()
	plt.savefig(path + "rewardhist")
	plt.close()

def graph_diff(diff,path,which,what,abs_diff=False,axes = None):
	""" 
		Hist of difference in AUC of [which] curve
		with same parameters in different 
		conditions
		example sim1 - mod true
				sim2 - mod false
		which - "Reward" or "Learning"
		what - which comparison is being done
			 - see function get_settings()
	"""

	comparison,base,c1,c2 = get_settings(what)

	plt.rcParams.update({'font.size': 22})
	fig, ax = plt.subplots(figsize=(6, 5))
	ax.hist(diff,color = c1)
	plt.ylabel("frequency")
	plt.xlabel(comparison)
	plt.title("%s" %which)
	plt.tight_layout()
	if abs_diff:
		nm = "diffabs_"
	else:
		nm = "diff_"
	plt.savefig(path + nm + which)
	plt.close()

def auc_by_alpha(auc,alphas,path,which):
	""" 
		Hist of difference in AUC of [which] curve
		with same parameters in different 
		conditions
		example sim - mod true
		which - "Reward" or "Learning"
		what - which comparison is being done
			 - see function get_settings()
	"""

	mina = min(alphas)
	maxa = max(alphas)
	alpha_list = np.unique(alphas)
	mean_auc = np.zeros(np.size(alpha_list))
	sem_auc = np.zeros(np.size(alpha_list))
	for idx,ele in enumerate(alpha_list):
		findme = np.where(alphas == ele)[0] 
		aucs = auc[findme]
		mean_auc[idx] = np.mean(aucs)
		sem_auc[idx] = stats.sem(aucs)

	plt.rcParams.update({'font.size': 22})
	fig, ax = plt.subplots(figsize=(6, 5))
	ax.errorbar(alpha_list,mean_auc,yerr=sem_auc)
	plt.ylabel("AUC")
	plt.xlabel("alpha actor")
	plt.title("%s" %which)
	plt.tight_layout()
	saveme = path + "auc_by_alphaA_" + which
	os.makedirs(saveme, exist_ok=True) 
	plt.savefig(saveme)
	plt.close()

# colormaps
def mapme(boolme):
	# pass a boolean and map to the appropriate colors
	# map = np.zeros((len(boolme),3))
	# for idx,boo in enumerate(boolme):
	# 	if boo == 0:
	# 		gray = 220/255
	# 		map[idx,:] = [gray,gray,gray]
	# 	elif boo == 1:
	# 		# blue
	# 		map[idx,:] = np.array([86.,187.,255.])/255.
	# 	elif boo == 2:
	# 		# red
	# 		map[idx,:] = np.array([255.,86.,103.])/255.
	# print("printing map")
	# print(map)
	colors = np.array(["gray", "red", "blue"])
	map = colors[boolme] 
	return map

def graph_diff_by_AUC(auc,diff,colors,titles,saveas,\
	maps,path,which,what,color_orig=None,abs_diff=False,axes = None):
	""" 
		Diff in AUC of [which] curve
		with same parameters in different 
		conditions, plotted by AUC of base 
		comparison (ex. unmodulated)
		- auc: base
		- diff: 

		Also plot and saved recolor by specified
		- colors/titles/saveas/maps
		
	"""

	comparison,base,c1,c2 = get_settings(what)
	cor, p = stats.spearmanr(auc, diff)
	s=5 # marker size
	sizes = np.zeros(len(auc)) + s

	# horizontal line at zero info
	lwidth = 2
	xs = np.linspace(np.min(auc),np.max(auc),200)
	h_line = np.array([0 for i in np.arange(len(xs))])
	plt.rcParams.update({'font.size': 22})

	# scatter uncolored
	fig, ax = plt.subplots(figsize=(6, 5))
	plt.plot(xs, h_line, 'lightgray', linewidth=lwidth) 
	if color_orig is not None:
		# make origin params larger
		idx = np.where(np.array(color_orig) >= 1)[0] # can be 1 or 2
		sizes[idx] = 100
		plt.rcParams.update({'font.size': 22})

		# scatter color, highlighting original params
		# mapp = ax.scatter(auc,diff,c = np.array(color_orig),cmap = "seismic",s=sizes,zorder=10)
		mapp = ax.scatter(auc,diff,c = mapme(color_orig),s=sizes)
		# plt.colorbar(mapp)
	else:
		ax.scatter(auc,diff,s=sizes,zorder=10,c="gray")

	plt.ylabel(comparison)
	plt.xlabel(base)
	plt.tight_layout()
	if abs_diff:
		nm = "diffabs_by_auc"
	else:
		nm = "diff_by_auc"
	plt.savefig(path + nm + which)
	plt.title("rho: %f p-val: %f" %(cor,p))
	plt.savefig(path + "diff_by_auc_withcorr" + which)
	plt.close()

	# iterate all the specified colors
	if axes is None :
		for idx,color in enumerate(colors):
			fig, ax = plt.subplots()
			plt.plot(xs, h_line, 'lightgray', linewidth=lwidth) 
			mapp = ax.scatter(auc,diff,c = np.array(color),cmap = maps[idx],s=sizes,zorder=10)
			plt.colorbar(mapp)
			plt.ylabel(comparison)
			plt.xlabel(base)
			plt.title(titles[idx])
			plt.tight_layout()
			plt.savefig(path + saveas[idx] + which)
			plt.close()