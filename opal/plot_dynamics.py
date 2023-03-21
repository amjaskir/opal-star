# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank

# Name: plot_dynamics.py
# Description: Aids to plot model dynamics of OpAL model overtime

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import metrics

def avg_sm(states,n_trials,n_states,C,axs,color,return_auc=False,print_auc=False,opacity=1.,linestyle="solid"):
	""" plot average softmax probability of selecting the 
	option, C, for each state 
	
	Inputs:
	- (list of) OpAL state(s)
	- n_trials
	- n_states
	- C, plotting average softmax of choice C
	- axs, figure axis to plot on
	- color, RGB or string for line color
	- print_auc, whether to print auc to output"""
	first = True
	for state in states:
		if first:
			sms = state.SM[:,C]
			first = False
		else:
			sms = np.vstack([sms,state.SM[:,C]])

	# average over states for trial-by-trial performance
	# plot
	s = 0.01
	linewidth = 2.
	xaxis = range(n_trials)
	if n_states > 1:
		avg_sm = np.mean(sms, axis = 0)
		sem_sm = stats.sem(sms, axis = 0)

		# shorten to specified number of trials
		avg_sm = avg_sm[0:n_trials]
		sem_sm = sem_sm[0:n_trials]

		axs.errorbar(xaxis,avg_sm,yerr=sem_sm, c = color, linewidth = linewidth, alpha=opacity,linestyle=linestyle)
		if return_auc:
			this_auc = metrics.auc(xaxis,avg_sm)
			return this_auc
		elif print_auc:
			this_auc = metrics.auc(xaxis,avg_sm)
			print("%s AUC: %.3f" %(color,this_auc))
	else:
		axs.errorbar(xaxis,sms[0:n_trials], c = color, linewidth = linewidth,alpha=opacity,linestyle=linestyle)

def avg_RT(states,R0,B,axs=None):
	# RT takes the form RT(a) ~ R0 + B/(1 + e^(Act(a)))
	# RT to start not RT to retrieve reward
	mRT = []
	for state in states:
		choices = state.C
		choiceAct = [state.Act[idx,c] for idx,c in enumerate(choices)]
		RTs = R0 + B/(1. + np.exp(choiceAct))
		mRT = np.append(mRT,np.mean(RTs))
	print(np.mean(mRT))


def avg_choice(states,n_trials,n_states,C,axs,color,return_auc=False,print_auc=False,chunk=False,n=6,linestyle="solid"):
	""" plot average softmax probability of selecting the 
	option, C, for each state 
	
	Inputs:
	- (list of) OpAL state(s)
	- n_trials
	- n_states
	- C, plotting average softmax of choice C
	- axs, figure axis to plot on
	- color, RGB or string for line color
	- print_auc, whether to print auc to output"""
	first = True
	for state in states:
		if first:
			sms = 1 - state.C
			if chunk: # average over chunks
				sms = [np.mean(sms[i:i + (n-1)]) for i in range(0, len(sms), n-1)] 
			first = False
		else:
			addme = 1 - state.C
			if chunk: # average over chunks
				addme = [np.mean(addme[i:i + (n-1)]) for i in range(0, len(addme), n-1)] 
			sms = np.vstack([sms, addme])

	# average over states for trial-by-trial performance
	# plot
	s = 0.01
	linewidth = 2.
	if chunk:
		n_trials = n
	xaxis = range(n_trials)
	if n_states > 1:
		avg_sm = np.mean(sms, axis = 0)
		sem_sm = stats.sem(sms, axis = 0)

		# shorten to specified number of trials
		avg_sm = avg_sm[0:n_trials]
		sem_sm = sem_sm[0:n_trials]

		axs.errorbar(xaxis,avg_sm,yerr=sem_sm, c = color, linewidth = linewidth,capsize = 7., linestyle=linestyle)
		if return_auc:
			this_auc = metrics.auc(xaxis,avg_sm)
			return this_auc
		elif print_auc:
			this_auc = metrics.auc(xaxis,avg_sm)
			print("%s AUC: %.3f" %(color,this_auc))
	else:
		axs.errorbar(xaxis,sms[0:n_trials], c = color, linewidth = linewidth, capsize = 7., linestyle=linestyle)

def avg_choice_gamble(states,n_trials,n_states,C,axs,color,return_auc=False,print_auc=False,chunk=False,n=6,opacity=1.0):
	""" plot average softmax probability of selecting the 
	option, C, for each state 
	
	Inputs:
	- (list of) OpAL state(s)
	- n_trials
	- n_states
	- C, plotting average softmax of choice C
	- axs, figure axis to plot on
	- color, RGB or string for line color
	- print_auc, whether to print auc to output"""
	first = True
	for state in states:
		if first:
			sms = state.C
			if chunk: # average over chunks
				sms = [np.mean(sms[i:i + (n-1)]) for i in range(0, len(sms), n-1)] 
			first = False
		else:
			addme = state.C
			if chunk: # average over chunks
				addme = [np.mean(addme[i:i + (n-1)]) for i in range(0, len(addme), n-1)] 
			sms = np.vstack([sms, addme])

	# average over states for trial-by-trial performance
	# plot
	s = 0.01
	linewidth = 2.
	if chunk:
		n_trials = n
	xaxis = range(n_trials)
	if n_states > 1:
		avg_sm = np.mean(sms, axis = 0)
		sem_sm = stats.sem(sms, axis = 0)

		# shorten to specified number of trials
		avg_sm = avg_sm[0:n_trials]
		sem_sm = sem_sm[0:n_trials]

		axs.errorbar(xaxis,avg_sm, c = color, linewidth = linewidth, alpha=opacity)
		if return_auc:
			this_auc = metrics.auc(xaxis,avg_sm)
			return this_auc
		elif print_auc:
			this_auc = metrics.auc(xaxis,avg_sm)
			print("%s AUC: %.3f" %(color,this_auc))
	else:
		axs.errorbar(xaxis,sms[0:n_trials], c = color, linewidth = linewidth,alpha=opacity)

def avg_sm_gamble(states,n_trials,n_states,axs,color,opacity=1.):
	""" plot average softmax probability of selecting the 
	option, C, for each state 
	
	Inputs:
	- (list of) OpAL state(s)
	- n_trials
	- n_states
	- C, plotting average softmax of choice C
	- axs, figure axis to plot on
	- color, RGB or string for line color"""
	first = True
	for state in states:
		high = state.probs[0] > .5
		# collect resoponses
		if first:
			if high:
				sms = state.SM[:,0]
			else:
				# optimal is not to gamble
				sms = 1 - state.SM[:,0]
			first = False
		else:
			if high:
				nxt = state.SM[:,0]
			else:
				# optimal is not to gamble
				nxt = 1 - state.SM[:,0]
			sms = np.vstack([sms,nxt])

	# average over states for trial-by-trial performance
	# plot
	s = 0.01
	linewidth = 2.
	xaxis = range(n_trials)
	if n_states > 1:
		avg_sm = np.mean(sms, axis = 0)
		sem_sm = stats.sem(sms, axis = 0)
		axs.errorbar(xaxis,avg_sm,yerr=sem_sm, c = color, linewidth = linewidth, alpha = opacity)
	else:
		axs.errorbar(xaxis,sms, c = color, linewidth = linewidth)


def avg_qs_dynamics(states,n_trials,n_states,C,axs,color_mag,rho=0.0,beta=1):
	""" plot avergage g and n actor weights over time 
	on specified axis for choice C 

	Inputs:
	- (list of) OpAL state(s)
	- n_trials
	- n_states
	- C, plotting actor evolution of choice C
	- axs, 2-element list, figure axis to plot on
	- color_mag, float, strength of red and green"""
	first = True
	for i in range(n_states):
		state = states[i]
		if first:
			Gs = state.QG[:,C]
			Ns = state.QN[:,C]
			Acts = beta*(1+rho)*state.QG[:,C] - beta*(1-rho)*state.QN[:,C]
			first = False
		else:
			Gs = np.vstack([Gs,state.QG[:,C]])
			Ns = np.vstack([Ns,state.QN[:,C]])
			this_act = beta*(1+rho)*state.QG[:,C] - beta*(1-rho)*state.QN[:,C]
			Acts = np.vstack([Acts,this_act])

	s = 0.01
	linewidth = 2.
	xaxis = np.arange(0,n_trials+1)
	if n_states > 1:
		mean_gs = np.mean(Gs,axis = 0)
		mean_ns = np.mean(Ns,axis = 0)
		mean_as = np.mean(Acts,axis = 0)
		sem_gs = stats.sem(Gs,axis = 0)
		sem_ns = stats.sem(Ns,axis = 0)
		sem_as = stats.sem(Acts,axis = 0)

		axs[0].errorbar(xaxis,mean_gs,yerr=sem_gs, c = (0.0,color_mag,0.0))
		axs[1].errorbar(xaxis,mean_ns,yerr=sem_ns, c = (color_mag,0.0,0.0))
		
		if color_mag == 0.5:
			axs[2].errorbar(xaxis,mean_as,yerr=sem_as, c = "yellow")
		else:
			axs[2].errorbar(xaxis,mean_as,yerr=sem_as, c = (color_mag,color_mag,color_mag))
	else:
		axs[0].errorbar(xaxis,Gs,color = (0.0,color_mag,0.0))
		axs[1].errorbar(xaxis,Ns,color = (color_mag,0.0,0.0))
		axs[2].errorbar(xaxis,Acts, color = (color_mag,color_mag,color_mag))

def avg_qs(states,n_trials,n_states,C,axs,color_mag,alpha=1.0):
	""" plot avergage g and n actor weights over time 
	on specified axis for choice C 

	Inputs:
	- (list of) OpAL state(s)
	- n_trials
	- n_states
	- C, plotting actor evolution of choice C
	- axs, 2-element list, figure axis to plot on
	- color_mag, float, strength of red and green"""
	first = True
	for i in range(n_states):
		state = states[i]
		if first:
			Gs = state.QG[:,C]
			Ns = state.QN[:,C]
			first = False
		else:
			Gs = np.vstack([Gs,state.QG[:,C]])
			Ns = np.vstack([Ns,state.QN[:,C]])

	s = 0.01
	linewidth = 2.
	xaxis = np.arange(0,n_trials+1)
	if n_states > 1:
		mean_gs = np.mean(Gs,axis = 0)
		mean_ns = np.mean(Ns,axis = 0)
		sem_gs = stats.sem(Gs,axis = 0)
		sem_ns = stats.sem(Ns,axis = 0)

		#subset by n_trials I want
		mean_gs = mean_gs[0:n_trials+1]
		mean_ns = mean_ns[0:n_trials+1]
		sem_gs = sem_gs[0:n_trials+1]
		sem_ns = sem_ns[0:n_trials+1]

		axs[0].errorbar(xaxis,mean_gs,yerr=sem_gs, c = (0.0,color_mag,0.0,alpha))
		axs[1].errorbar(xaxis,mean_ns,yerr=sem_ns, c = (color_mag,0.0,0.0,alpha))
	else:
		axs[0].errorbar(xaxis,Gs[0:n_trials+1],color = (0.0,color_mag,0.0,alpha))
		axs[1].errorbar(xaxis,Ns[0:n_trials+1],color = (color_mag,0.0,0.0,alpha))

def plt_CR(state,n_trials,ax):
	"""
	Plot the C (choice) and R (reward) of the trial
	"""
	t = np.arange(n_trials)
	ax.scatter(t,(state[0].C[0:n_trials] == 0), color = "black")
	ax.scatter(t,state[0].R[0:n_trials]*1.15, color = "green")

def avg_q(states,n_trials,n_states,C,axs):
	""" plot avergage Q value in standard rl model

	Inputs:
	- (list of) OpAL state(s)
	- n_trials
	- n_states
	- C, plotting actor evolution of choice C
	- axs, 2-element list, figure axis to plot on
	- color_mag, float, strength of red and green"""
	first = True
	for i in range(n_states):
		state = states[i]
		if first:
			Qs = state.Q[:,C]
			first = False
		else:
			Qs = np.vstack([Qs,state.Q[:,C]])

	s = 0.01
	linewidth = 2.
	xaxis = np.arange(0,n_trials+1)
	if n_states > 1:
		mean_qs = np.mean(Qs,axis = 0)
		sem_qs = stats.sem(Qs,axis = 0)
		axs.errorbar(xaxis,mean_qs,yerr=sem_qs)
	else:
		axs.errorbar(xaxis,Qs)

def avg_convex(states,n_trials,n_states,axs):
	""" plot avergage inequality between diff in G and N
	weights between two options. Assumes first option has highest
	probability of reward

	Inputs:
	- (list of) OpAL state(s)
	- n_trials
	- n_states
	- axs, figure axis to plot on"""
	first = True
	for i in range(n_states):
		state = states[i]
		if first:
			Gs = state.QG[:,0] - state.QG[:,1]
			Ns = state.QN[:,0] - state.QN[:,1]
			first = False
		else:
			Gs = np.vstack([Gs,state.QG[:,0] - state.QG[:,1]])
			Ns = np.vstack([Ns,state.QN[:,0] - state.QN[:,1]])

	s = 0.01
	linewidth = 2.
	xaxis = np.arange(0,n_trials+1)
	if n_states > 1:
		mean_gs = np.mean(Gs,axis = 0)
		mean_ns = np.mean(Ns,axis = 0)
		sem_gs = stats.sem(Gs,axis = 0)
		sem_ns = stats.sem(Ns,axis = 0)

		axs.errorbar(xaxis,mean_gs,yerr=sem_gs, c = "green")
		axs.errorbar(xaxis,mean_ns,yerr=sem_ns, c = "red")
	else:
		axs.errorbar(xaxis,Gs,color = "green")
		axs.errorbar(xaxis,Ns,color = "red")

def avg_vs(states,n_trials,n_states,crit,C,axs,color_mag):
	""" plot avergage critic value weights over time 
	on specified axis for choice C 

	Inputs:
	- (list of) OpAL state(s)
	- n_trials
	- n_states
	- crit, type of critic
	- C, plotting actor evolution of choice C if critic is SA
	- axs, figure axis to plot on
	- color_mag, float, strength grey for plotting"""
	first = True
	if crit == "S":
		for state in states:
			if first:
				Vs = state.V
				first = False
			else:
				Vs = np.vstack([Vs,state.V])
	elif crit == "Bayes-SA":
		for state in states:
			if first:
				Vs = state.mean
				first = False
			else:
				Vs = np.vstack([Vs,state.mean])
	else:
		for state in states:
			if first:
				Vs = state.V[:,C]
				first = False
			else:
				Vs = np.vstack([Vs,state.V[:,C]])
	
	s = 0.01
	linewidth = 2.
	xaxis = np.arange(0,n_trials+1)
	if crit == "Bayes-SA":
		xaxis = np.arange(0,n_trials)
	if n_states > 1:
		mean_vs = np.mean(Vs,axis = 0)
		sem_vs = stats.sem(Vs,axis = 0)
		axs.errorbar(xaxis,mean_vs,yerr=sem_vs, c = (color_mag,color_mag,color_mag))
	else:
		axs.errorbar(xaxis,Vs,color = (color_mag,color_mag,color_mag), s = 10)

def avg_rho(states,n_trials,n_states,axs,color="purple"):
	"""plot evolution of average rho over time

	Inputs:
	- (list of) OpAL state(s)
	- n_trials
	- n_states
	- axs, figure axis to plot on"""
	first = True
	for state in states:
		if first:
			rhos = state.rho
			first = False
		else:
			rhos = np.vstack([rhos,state.rho])

	# plot
	s = 0.01
	linewidth = 2.
	xaxis = range(n_trials)
	if n_states > 1:
		# average over states for trial-by-trial performance
		avg_r = np.mean(rhos, axis = 0)
		sem_r = stats.sem(rhos, axis = 0)

		# subset by trials I want
		avg_r = avg_r[0:n_trials]
		sem_r = sem_r[0:n_trials]
		axs.errorbar(xaxis,avg_r,yerr=sem_r, c = color)
	else:
		axs.errorbar(xaxis,rhos[0:n_trials], c=color, linewidth=linewidth)


def avg_PE(states,n_trials,n_states,axs):
	"""plot evolution of average PE over time 

	Inputs:
	- (list of) OpAL state(s)
	- n_trials
	- n_states
	- axs, figure axis to plot on"""
	first = True
	for state in states:
		if first:
			PEs = state.PE
			first = False
		else:
			PEs = np.vstack([PEs,state.PE])

	# plot
	s = 0.01
	linewidth = 2.
	xaxis = range(n_trials)
	if n_states > 1:
		# average over states for trial-by-trial performance
		avg_PE = np.mean(PEs, axis = 0)
		sem_PE = stats.sem(PEs, axis = 0)
		axs.errorbar(xaxis,avg_PE,yerr=sem_PE, c = "purple")
	else:
		axs.errorbar(xaxis,PEs, c="purple", linewidth=linewidth)