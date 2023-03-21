# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank

# Name: opal.py
# Description: Opponent Actor Learner model from Collins and Frank (2014)
# One actor accumulates evidence for benefits of an action
# The other, the evidence for the cost of the same action. 
# DA at choice moderates contribution of actors to action selection

import numpy as np
import sys
import random
import copy
from scipy.stats import beta

class OpAL:
	def __init__(self, n_trials, crit, v0, n_options, probs, r_mag, l_mag,\
		anneal=False, use_var=False, pgrad=False, T=100.0,trial_type = None, norm_type = None):
		""" Tracks and updates information for learning via OpAL 
		in a generated state.

		Inputs:
		- n_trials (int), the number of trials for simulation in state
		- crit (str), whether the critic is tracking "S" or "SA" value
		- v0 (flt), intial value of the critic
		- n_options (int), number of possible actions
		- probs (1-d np.array or flt), probability of reward for selecting each action
		- r_mag (np.array or flt), reward magnitude
		- l_mag (np.array or flt), loss magnitude
		"""
		
		def init_crit(crit):
			""" Helper to intialize critic
			"""
			if crit == "S":		# critic value, V(s)	
				return np.zeros(sz+1) + v0	
			elif crit == "S-ctx":	# for palmineteri
				return np.zeros((sz+1, int(n_options/2))) + v0					
			elif crit == "SA":  # critic value, V(s,a)
				return np.zeros((sz+1, n_options)) + v0	
			elif crit == "Bayes":
				return np.ones((sz+1,2))   # beta func: alpha (reward count), beta (no reward count)	
			elif crit == "Bayes-SA":
				base = np.ones((sz+1,n_options*2))
				base[0,0:(n_options*2):2] = 1 + v0[0] # add any priors to first V
				base[0,1:(n_options*2):2] = 1 + v0[1]
				return base   # beta func: alpha (reward count), beta (no reward count)per action
			else:
				err = 'Invalid value given for arg crit. %s given' %crit
				raise Exception(err)

		sz = n_trials
		self.n_trials = sz
		self.n_options = n_options					# number of choices
		self.crit = crit 							# critic type
		self.V = init_crit(crit)					# critic
		self.QG = np.zeros((sz+1, n_options)) + 1	# Go Actor Values
		self.QN = np.zeros((sz+1, n_options)) + 1	# NoGo Actor Values
		self.alphags = np.zeros(sz)					# Go Actor Learning Rate
		self.alphans = np.zeros(sz)					# NoGo Actor Learning Rate
		self.beta_g = np.zeros(sz)					# Inverse temp, Go Actor
		self.beta_n = np.zeros(sz)					# Inverse temp, NoGo Actor
		self.Act = np.zeros((sz, n_options))		# Act values for softmax
		self.H = np.zeros(sz) 						# entropy
		self.gammas = np.zeros(sz+1)				# decay to prior gammas
		self.SM  = np.zeros((sz, n_options))		# Softmax values
		self.rho = np.zeros(sz)						# DA at choice
		self.C = np.zeros(sz,dtype=np.int)			# choice
		self.R = np.zeros(sz,dtype=np.int)			# indicator variable of reward
		self.beta_dist = np.zeros((sz+1,2)) +1		# beta distribution of avg reward
		self.probs = probs							# prob of reward for choice
		self.r_mag = r_mag							# magnitude of reward for choice
		self.l_mag = l_mag							# magnitude of loss for choice
		self.PE  = np.zeros(sz)						# choice PE
		self.idx = 0								# idx of trial in state 
		self.anneal = anneal						# anneal the learning rate
		self.use_var = use_var						# use var to adjust annealing
		self.pgrad = pgrad
		self.norm_type = norm_type					# anneal the learning rate
		self.T = T 									# annealing parameter 
		self.counts = np.zeros(n_options)			# count of when each option has been chose
		# NOTE - self.params also available in learning.py script

		### Simulation specific elements ###
		# Bayesian values
		self.mean = np.zeros(sz+1)	
		self.std = np.zeros(sz+1)
		self.var = np.zeros(sz+1)

		# initialize first trial accordingly
		# TODO: if not a naive prior, need to update with appropriate critic V
		mean, var = beta.stats(self.beta_dist[0,0],self.beta_dist[0,1],moments='mv')
		self.mean[0] = mean
		self.var[0] = var
		self.std[0] = np.sqrt(var)

		# Inference paradigm
		self.prior = np.zeros((sz+1,2)) + .5		# probability high or low state
													# initialized to chance
		# Rutledge data
		self.trial_type = trial_type				# 0-gain,1-loss-2-mixed
		self.sure_mags  = np.zeros(sz) 				# mag of sure thing
		self.r_mags	    = np.zeros(sz) 				# gain, relative to sure
		self.l_mags		= np.zeros(sz)				# loss, relative to sure
		self.multiplier = np.zeros(sz)				# multiplier applied to sure
		self.advantage 	= np.zeros(sz)				# advantage of gamble on average over sure
		self.offer_V    = np.zeros(sz+1)			# EV of current offer
		self.V_tr    	= np.zeros(sz+1)			# Learned avg gamble value 
		self.PE_tr		= np.zeros(sz)				# PE according to V_tr

		# Palminteri
		self.ctx = None								# context of actions
		self.ctx_order = []							# ctx order
		self.C_in_ctx = np.zeros(sz,dtype=np.int)	# context choice
		self.Q = np.zeros((sz+1, n_options)) + v0	# for relative model, standard Q value								

	def policy (self,thresh=0,forced_actions=None,forced_rewards=None,state_idx=None):
		"""
		Selects an action via softmax. 
		Activation function is a linear combination of the two actors 
		according to betaG/betaN, asymmetry determined by tonic DA
		"""
		idx = self.idx 				# internal time index of state
		probs = self.probs			# prob of reward for an action
		beta_g = self.beta_g[idx]	# inverse temp for go action value	
		beta_n = self.beta_n[idx]	# inverse temp for nogo action value
		crit  = self.crit    		# critic type

		# calc Act thalamus activation
		Act = beta_g*self.QG[idx] - beta_n*self.QN[idx]
		self.Act[idx,:] = Act

		if idx < thresh:
			# random policy
			ps = np.ones(self.n_options)/self.n_options
		else:
			# multioption softmax (invariant to constant offsets)
			newAct = Act - np.max(Act)
			expAct = np.exp(newAct)
			ps = expAct/np.sum(expAct)
		self.SM[idx,:] = ps
		self.H[idx] = -np.sum(ps*np.log2(ps))
		if np.isnan(self.H[idx]): #deterministic, one action has 100% probability of being selected
			self.H[idx] = 0
		cs_ps = np.cumsum(ps) # cumulative sum of probabilities

		# random exploration?
		epsilon = 0	# add epislon exploration to softmax?
		if epsilon > 0:
			random_explore = (np.random.random_sample() < epsilon)
			if random_explore:
				ps = [1/self.n_options for x in ps]
				cs_ps = np.cumsum(ps)

		# select action
		if forced_actions is None:
			sample = np.random.random_sample()
			selected = False
			check = 0
			while not selected:
				if sample < cs_ps[check]:
					C = check
					selected = True
				else:
					check = check + 1
		else:
			C = forced_actions[state_idx,idx]
		self.C[idx] = C
		self.counts[C] = self.counts[C] + 1

		# decide whether a reward is delivered
		if forced_rewards is None:
			reward = np.random.binomial(size=1, n=1, p= probs[C])[0]
		else:
			reward = forced_rewards[state_idx,idx]
		self.R[idx] = reward # indicator that reward was received
		if reward == 0:
			reward = self.l_mag[C]
		else:
			reward = self.r_mag[C]

		# calculate PE
		if crit == "S":
			PE = reward - self.V[idx]
		elif crit == "S-ctx":
			PE = reward - self.V[idx,self.ctx_order[idx]]
		elif crit == "SA":
			PE = reward - self.V[idx,C]
		elif crit == "Bayes":
			b_dist = self.V[idx,:]
			mean, var = beta.stats(b_dist[0],b_dist[1],moments='mv')
			# assume rmag and lmag are same for both options
			infer_val = self.r_mag[0]*mean + self.l_mag[0]*(1-mean)
			PE = reward - infer_val 
		elif crit == "Bayes-SA":
			choice_idx = C*2
			b_dist_a = self.V[idx,choice_idx]
			b_dist_b = self.V[idx,choice_idx+1]
			mean, var = beta.stats(b_dist_a,b_dist_b,moments='mv')
			# assume rmag and lmag are same for both options
			infer_val = self.r_mag[0]*mean + self.l_mag[0]*(1-mean)
			PE = reward - infer_val 
		else:
			err = 'Invalid value given for arg crit. %s given' %crit
			raise Exception(err)
		self.PE[idx] = PE

	def critic (self,alpha,gamble=False,decay=False,gamma=1.0):
		""" Updates value of critic for chosen action
		"""
		idx = self.idx
		C = self.C[idx]		# choice
		PE = self.PE[idx]	# choice PE
		crit = self.crit
		anneal = self.anneal
		T = self.T

		# anneal the learning rate
		# if anneal:
		# 	alpha = alpha/(1. + self.counts[C]/T)

		if crit == "S":
			if gamble:
				if C == 0: # didn't gamble, carry over values, no update
					self.V[idx+1] = self.V[idx]
					self.beta_dist[idx+1] = self.beta_dist[idx]
					self.mean[idx+1] = self.mean[idx]
					self.var[idx+1] = self.var[idx]
					self.std[idx+1] = self.std[idx]
					return
			
			# update
			self.V[idx+1] = self.V[idx] + alpha*PE

			# increase alpha/beta counts of separate dist
			# no Bayesian critic, use beta dist over avg reward of all actions
			R = self.R[idx]
			self.beta_dist[idx+1,0] = self.beta_dist[idx,0] + R
			self.beta_dist[idx+1,1] = self.beta_dist[idx,1] + (1-R)
			alph = self.beta_dist[idx+1,0]/self.n_options
			bet = self.beta_dist[idx+1,1]/self.n_options
		elif crit == "S-ctx":
			self.V[idx+1] = self.V[idx]	# carry over values for all choices
			self.V[idx+1,self.ctx_order[idx]] = self.V[idx,self.ctx_order[idx]] + alpha*PE

			# increase alpha/beta counts of separate dist
			# no Bayesian critic, use beta dist over avg reward of all actions
			R = self.R[idx]
			self.beta_dist[idx+1,0] = self.beta_dist[idx,0] + R
			self.beta_dist[idx+1,1] = self.beta_dist[idx,1] + (1-R)
			alph = self.beta_dist[idx+1,0]/self.n_options
			bet = self.beta_dist[idx+1,1]/self.n_options
		elif crit == "SA":
			self.V[idx+1] = self.V[idx]	# carry over values for all choices
			self.V[idx+1,C] = self.V[idx,C] + alpha*PE  # update chosen

			# increase alpha/beta counts of separate dist
			# no Bayesian critic, use beta dist over avg reward of all actions
			R = self.R[idx]
			self.beta_dist[idx+1,0] = self.beta_dist[idx,0] + R
			self.beta_dist[idx+1,1] = self.beta_dist[idx,1] + (1-R)
			alph = self.beta_dist[idx+1,0]/self.n_options
			bet = self.beta_dist[idx+1,1]/self.n_options
		elif crit == "Bayes":
			# increase alpha/beta counts of critic
			# A bayesian critic which doesn't care about actions, 
			# can just update the critic directly
			R = self.R[idx]
			self.V[idx+1,0] = self.V[idx,0] + R
			self.V[idx+1,1] = self.V[idx,1] + (1-R)
			alph = self.V[idx+1,0]/self.n_options
			bet = self.V[idx+1,1]/self.n_options
		elif crit == "Bayes-SA":
			# handle gambling things
			if gamble:
				if C == 0: # didn't gamble, carry over values, no update
					self.V[idx+1] = self.V[idx]
					return
				else:
					C = 0 # only one option...this is bad coding, sorry
			choice_idx = C*2

			# increase alpha/beta counts of critic
			R = self.R[idx]
			self.V[idx+1] = self.V[idx] # carry over values for all choices
			self.V[idx+1,choice_idx] = self.V[idx,choice_idx] + R
			self.V[idx+1,choice_idx+1] = self.V[idx,choice_idx+1] + (1-R)

			# update mean for next trial
			# averages alphas and betas across options
			alph = np.mean(self.V[idx+1,::2])
			bet = np.mean(self.V[idx+1,1::2])
		else:
			err = 'Invalid value given for arg crit. %s given' %crit
			raise Exception(err)

		# set mean, var, and std of state
		mean, var = beta.stats(alph,bet,moments='mv')
		self.mean[idx+1] = mean
		self.var[idx+1] = var
		self.std[idx+1] = np.sqrt(var)

		if decay:
			# print("decaying now")
			# print("chosen: %d" %C)
			for i in np.arange(self.n_options):
				if not (i == C):
					# decay unchosen 
					choice_idx = i*2
					# print("decaying %d" %i)
					# print(self.V[idx+1,choice_idx])
					# print(self.V[idx+1,choice_idx+1])
					self.V[idx+1,choice_idx] = self.V[idx+1,choice_idx]*gamma # alpha
					self.V[idx+1,choice_idx+1] = self.V[idx+1,choice_idx+1]*gamma # beta
					# print(self.V[idx+1,choice_idx])
					# print(self.V[idx+1,choice_idx+1])


	def act (self,alpha,norm=False,mag=1,var=None,\
		hebb=True,bound=False,lim=5,gamble=False):
		"""
		Updates the Q vales for the direct (G) and indirect (NG) actors
		for the next time step for chosen action

		norm - T/F normalize PE by tanh and provided mag
		mag - normaliziation factor, normally max mag of potential reward
		hebb - T/F use 3-factor hebbian update
		full - full information, model gets feedback for every action
			   even if it was not selected, does not overwrite G/N
		"""
		idx = self.idx
		PE = self.PE[idx]
		C = self.C[idx]	# choice
		anneal = self.anneal
		use_var = self.use_var
		pgrad = self.pgrad
		T = self.T

		self.QG[idx+1] = self.QG[idx] #carry over values
		self.QN[idx+1] = self.QN[idx] 

		if gamble:
			if C == 1: # took the gamble
				C = 0  # choice index
			else:
				return

		# Normalizes PE between -1 and 1 by "memory" of largest recent
		# magnitude reward
		if norm:
			if self.norm_type is None:
				PE = PE/mag 
			elif self.norm_type == "tanh":
				PE = np.tanh(PE)
			else:
				err = 'Norm type %s is not supported' %(self.norm_type)
				raise Exception(err)

		# set alphaG and alphaN
		if var == "lrate":
			# TODO: THIS IS INCORRECT
			err = 'lrate no longer supported'
			raise Exception(err)
		
			rho = self.rho[idx]
			alphag = alpha*(1. + rho) # rho is in [-1,1]
			alphan = alpha*(1. - rho)
			
			# constrain between [0,1]
			alphag = np.max([0,np.min([alphag,1])])
			alphan = np.max([0,np.min([alphan,1])])
		else:
			alphag = alpha
			alphan = alpha

		# anneal the learning rate
		if anneal:
			if use_var:
				alphag = alphag/(1. + 1/(self.var[idx]*T*10))  # for T = 50, trial 1 0.1/T*var =  .024, similar to 1/50
				alphan = alphan/(1. + 1/(self.var[idx]*T*10)) 
			else:
				alphag = alphag/(1. + idx/T)
				alphan = alphan/(1. + idx/T)
			# if self.crit == "Bayes-SA" or self.crit == "Bayes":  # if using bayes critic, use uncertainty for annealing
			# 	alphag = alphag/(1. + 1/(self.var[idx]*T*10))  # for T = 50, trial 1 0.1/T*var =  .024, similar to 1/50
			# 	alphan = alphan/(1. + 1/(self.var[idx]*T*10))  
			# else:
			# 	alphag = alphag/(1. + idx/T)
			# 	alphan = alphan/(1. + idx/T)
		
		# #asymmetric learning from +/- PE
		# asym = True
		# if asym:
		# 	if PE > 0:
		# 		alphan = alphan*.99
		# 	else:
		# 		alphag = alphag*.99

		# save learning rates
		self.alphags[idx] = alphag
		self.alphans[idx] = alphan


		# use hebbian term? 
		if hebb:
			updateG = alphag*self.QG[idx,C]*PE
			updateN = alphan*self.QN[idx,C]*-PE
		else:
			updateG = alphag*PE
			updateN = alphan*-PE
		
		# use soft bound? 
		if bound:
			updateG = updateG*(lim - self.QG[idx,C])
			updateN = updateN*(lim - self.QN[idx,C])
			
                # use policy gradient? (simplifies to 1-p(C))  
		if pgrad: 
			ps= self.SM[idx]
			# use below if just one action
			#Act2 = beta_g - beta_n # baseline with QG=QN=1 for comparison in forced choice
			#expAct = np.exp(Act)
			#expAct2 = np.exp(Act2)
			#ps = expAct/(expAct + expAct2)
			updateG = updateG*(1. - ps[C])
			updateN = updateN*(1. - ps[C])

		# main update
		self.QG[idx+1,C] = self.QG[idx,C] + updateG
		self.QN[idx+1,C] = self.QN[idx,C] + updateN

		# actor values should not be negative, represent activations
		self.QG[idx+1,C] = np.max([0,self.QG[idx+1,C]])
		self.QN[idx+1,C] = np.max([0,self.QN[idx+1,C]])

	def update_other_actions(self,alpha_c,alpha_a,norm=False,mag=1,hebb=True,\
		var=None,bound=False,lim=5):
		"""
		Cycle through all unselected actions and update as if chosen.
		Call after standard criitc()/actor() functions are called
		with same arguments as they were provided

		Call after crit/act update of selected action

		choice_only: true   - only update critic with selected action
							- crit converges avg reward of policy
					 false  - is equivalent to forcing the 
							  agent to select each action each turn
							- critic converges to avg reward across actions
		"""
		idx = self.idx 		# current state index
		ihelp = idx + 1		# updated idx (treat as new idx)
		C = self.C[idx]		# action selected/already updated
		R = self.R[idx] 	# original reward from C
		PE = self.PE[idx]	# original PE from C

		# add a buffer if last trial
		the_end = (ihelp == self.n_trials)
		if the_end:
			self.C = np.append(self.C,np.array([0]))
			self.R = np.append(self.R,np.zeros(1))
			self.PE = np.append(self.PE,np.zeros(1))
			self.QG = np.vstack( (self.QG, np.empty(np.shape(self.QG[ihelp]))) )
			self.QN = np.vstack( (self.QN, np.empty(np.shape(self.QN[ihelp]))) )

			if (self.crit == "S"): 
				self.V = np.append(self.V,np.zeros(1))
			elif(self.crit == "SA"):
				self.V = np.vstack( (self.V, np.empty(np.shape(self.V[ihelp]))) )
			elif (self.crit == "Bayes"):
				pass
			elif (self.crit == "Bayes-SA"):
				pass
			else:
				err = "Crit %s not supported" %self.crit
				raise Exception(err)


		# determine which options to update
		n_options = self.n_options
		update = np.arange(n_options)
		update = np.delete(update,C) # remove chosen option from update

		# update all options for next time step
		# computed in place of array
		self.idx = ihelp
		for c in update:
			# updates entry in ihelp + 1
			self.policy_forced(c)
			self.act(alpha_a,norm=norm,mag=mag,hebb=hebb,var=var,bound=bound,lim=lim)
			self.critic(alpha_c)

			# update ihelp data with new info
			# critic
			if (self.crit == "S") or (self.crit == "SA"):
				self.V[ihelp] = self.V[ihelp+1]
			elif (self.crit == "Bayes"):
				pass
			elif (self.crit == "Bayes-SA"):
				pass
			else:
				err = "Crit %s not supported" %self.crit
				raise Exception(err)

			# actors
			self.QG[ihelp] = self.QG[ihelp+1]
			self.QN[ihelp] = self.QN[ihelp+1]

		# reset to master index
		self.idx = idx

		# remove buffer
		if the_end:
			self.C = np.delete(self.C,ihelp,0)
			self.R = np.delete(self.R,ihelp,0)
			self.PE = np.delete(self.PE,ihelp,0)
			self.QG = np.delete(self.QG,ihelp,0)
			self.QN = np.delete(self.QN,ihelp,0)

			if (self.crit == "S") or (self.crit == "SA"):
				self.V = np.delete(self.V,ihelp,0)
			elif (self.crit == "Bayes"):
				pass
			elif (self.crit == "Bayes-SA"):
				pass
			else:
				err = "Crit %s not supported" %self.crit
				raise Exception(err)


	######## Decay Add-ons #########
	def decay (self,gamma):
		"""
		Decays actor values. 
		Adapted from Moller and Bogacz, 2019
		"""
		idx = self.idx + 1 				# decay values for next trial
		self.QG[idx] = self.QG[idx] - gamma*self.QG[idx]
		self.QN[idx] = self.QN[idx] - gamma*self.QN[idx]

	def decay_to_prior (self,prior=1,gamma=1):
		"""
		Decays actor values to naive prior. 
		Modified from Franklin & Frank (2015)
		"""
		idx = self.idx + 1 				# decay values for next trial
		#ps = self.SM[idx-1]
		#H = -np.sum(ps*np.log2(ps))
		gamma = gamma + gamma*self.H[idx-1] # nick had intercept and slope of 5, slope should be negative such that higher volatility = more decay
		logit = 1/(1 + np.exp(-gamma))	# inverse logit
		self.gammas[idx] = logit
		self.QG[idx] = self.QG[idx]*logit + prior*(1 - logit)
		self.QN[idx] = self.QN[idx]*logit + prior*(1 - logit)

		# also decay bayesian critic if it exists
		if self.crit == "Bayes-SA" or self.crit == "Bayes":
			self.V[idx] = self.V[idx]*logit #assume zero prior
		else:
			self.beta_dist[idx] = self.beta_dist[idx]*logit #assume zero prior

	######## Act Variants #########
	def act_gamble (self,alpha,norm=False,mag=1,hebb=True):
		"""
		Updates the Q vales for the direct (G) and indirect (NG) actors
		for the next time step if gamble was selected
		"""
		idx = self.idx
		C = self.C[idx]	# choice
		PE = self.PE[idx]

		self.QG[idx+1] = self.QG[idx] #carry over values
		self.QN[idx+1] = self.QN[idx] 

		# Normalizes PE between -1 and 1 by "memory" of largest recent
		# magnitude reward
		if norm:
			PE = PE/mag #np.tanh(PE/mag)

		# took the gamble
		if C == 1:
			if hebb:
				self.QG[idx+1] = np.max([0,self.QG[idx] + alpha*self.QG[idx]*PE])
				self.QN[idx+1] = np.max([0,self.QN[idx] + alpha*self.QN[idx]*-PE])
			else:
				self.QG[idx+1] = np.max([0,self.QG[idx] + alpha*PE])
				self.QN[idx+1] = np.max([0,self.QN[idx] + alpha*-PE])
	
	######## Full information #########

	def counterfactual_update (self, C_alt, alpha_c, alpha_a, T, norm, mag):
		"""
			Updates G/N and critic as if other_action were selected.

			SM/Act/PE from policy not updated
		"""

		idx = self.idx
		probs = self.probs
		crit  = self.crit    
		ctx = self.ctx_order[idx]		
		
		# decide whether a reward is delivered ############################
		reward = np.random.binomial(size=1, n=1, p= probs[C_alt])[0]
		R = reward
		if reward == 0:
			reward = self.l_mag[C_alt]
		else:
			reward = self.r_mag[C_alt]

		# calculate PE
		if crit == "S":
			PE = reward - self.V[idx]
		elif crit == "SA":
			PE = reward - self.V[idx,C_alt]
		elif crit == "S-ctx":
			PE = reward - self.V[idx,ctx]
		else:
			err = 'Invalid value given for arg crit. %s given' %crit
			raise Exception(err)

		# Update critic ###################################################
		if crit == "S":
			self.V[idx+1] = self.V[idx] + alpha_c*PE

			# increase alpha/beta counts of separate dist
			# no Bayesian critic, use beta dist over avg reward of all actions
			R = self.R[idx]
			self.beta_dist[idx+1,0] = self.beta_dist[idx,0] + R
			self.beta_dist[idx+1,1] = self.beta_dist[idx,1] + (1-R)
			alph = self.beta_dist[idx+1,0]/self.n_options
			bet = self.beta_dist[idx+1,1]/self.n_options
		elif crit == "SA":
			self.V[idx+1,C_alt] = self.V[idx,C_alt] + alpha_c*PE  # update chosen

			# increase alpha/beta counts of separate dist
			# no Bayesian critic, use beta dist over avg reward of all actions
			self.beta_dist[idx+1,0] = self.beta_dist[idx,0] + R
			self.beta_dist[idx+1,1] = self.beta_dist[idx,1] + (1-R)
			alph = self.beta_dist[idx+1,0]/self.n_options
			bet = self.beta_dist[idx+1,1]/self.n_options
		elif crit == "S-ctx":
			self.V[idx+1,ctx] = self.V[idx,ctx] + alpha_c*PE  # update chosen

			# increase alpha/beta counts of separate dist
			# no Bayesian critic, use beta dist over avg reward of all actions
			self.beta_dist[idx+1,0] = self.beta_dist[idx,0] + R
			self.beta_dist[idx+1,1] = self.beta_dist[idx,1] + (1-R)
			alph = self.beta_dist[idx+1,0]/self.n_options
			bet = self.beta_dist[idx+1,1]/self.n_options
		else:
			err = 'Invalid value given for arg crit. %s given' %crit
			raise Exception(err)

		# set mean, var, and std of state
		mean, var = beta.stats(alph,bet,moments='mv')
		self.mean[idx+1] = mean
		self.var[idx+1] = var
		self.std[idx+1] = np.sqrt(var)

		# Update actors ###################################################
		if norm:
			if self.norm_type is None:
				PE = PE/mag 
			else:
				err = 'Norm type %s is not supported' %(self.norm_type)
				raise Exception(err)

		# calc learning rates
		alphag = alpha_a
		alphan = alpha_a
		
		# anneal with var
		alphag = alphag/(1. + 1/(self.var[idx]*T*10))  # for T = 50, trial 1 0.1/T*var =  .024, similar to 1/50
		alphan = alphan/(1. + 1/(self.var[idx]*T*10)) 

		# use hebb?
		updateG = alphag*self.QG[idx,C_alt]*PE
		updateN = alphan*self.QN[idx,C_alt]*-PE

		# main update
		self.QG[idx+1,C_alt] = self.QG[idx,C_alt] + updateG
		self.QN[idx+1,C_alt] = self.QN[idx,C_alt] + updateN

		# actor values should not be negative, represent activations
		self.QG[idx+1,C_alt] = np.max([0,self.QG[idx+1,C_alt]])
		self.QN[idx+1,C_alt] = np.max([0,self.QN[idx+1,C_alt]])

	######## Policy Variants #########
	def policy_subset (self,subset,all_actions,ctx, post = False):
		"""
		Selects an action via softmax, when only a subset of options are available
		Activation function is a linear combination of the two actors 
		according to betaG/betaN, asymmetry determined by tonic DA
		"""
		idx = self.idx 				# internal time index of state
		probs = self.probs			# prob of reward for an action
		beta_g = self.beta_g[idx]	# inverse temp for go action value	
		beta_n = self.beta_n[idx]	# inverse temp for nogo action value
		crit  = self.crit    		# critic type
	
		# only consider subset I care about
		Gs = self.QG[idx][subset]
		Ns = self.QN[idx][subset]
		complement = np.delete(all_actions,subset)

		# calc Act thalamus activation
		Act = beta_g*Gs - beta_n*Ns
		self.Act[idx,subset] = Act
		self.Act[idx,complement] = 0

		# multioption softmax (invariant to constant offsets)
		newAct = Act - np.max(Act)
		expAct = np.exp(newAct)
		ps = expAct/np.sum(expAct)
		self.SM[idx,subset] = ps
		self.SM[idx,complement] = 0
		
		self.H[idx] = -np.sum(ps*np.log2(ps))
		if np.isnan(self.H[idx]): #deterministic, one action has 100% probability of being selected
			self.H[idx] = 0
		cs_ps = np.cumsum(ps) # cumulative sum of probabilities

		# random exploration?
		epsilon = 0	# add epislon exploration to softmax?
		if epsilon > 0:
			random_explore = (np.random.random_sample() < epsilon)
			if random_explore:
				ps = [1/self.n_options for x in ps]
				cs_ps = np.cumsum(ps)

		# select action
		sample = np.random.random_sample()
		selected = False
		check = 0
		while not selected:
			if sample < cs_ps[check]:
				C = check
				selected = True
			else:
				check = check + 1
		self.C_in_ctx[idx] = C

		C = subset[C] # put in terms of 8 options
		self.C[idx] = C
		self.counts[C] = self.counts[C] + 1

		# decide whether a reward is delivered
		reward = np.random.binomial(size=1, n=1, p= probs[C])[0]
		self.R[idx] = reward # indicator that reward was received
		if reward == 0:
			reward = self.l_mag[C]
		else:
			reward = self.r_mag[C]

		# calculate PE
		if crit == "S":
			PE = reward - self.V[idx]
		elif crit == "S-ctx":
			PE = reward - self.V[idx,ctx]
		elif crit == "SA":
			PE = reward - self.V[idx,C]
		elif crit == "Bayes":
			b_dist = self.V[idx,:]
			mean, var = beta.stats(b_dist[0],b_dist[1],moments='mv')
			# assume rmag and lmag are same for both options
			infer_val = self.r_mag[0]*mean + self.l_mag[0]*(1-mean)
			PE = reward - infer_val 
		elif crit == "Bayes-SA":
			choice_idx = C*2
			b_dist_a = self.V[idx,choice_idx]
			b_dist_b = self.V[idx,choice_idx+1]
			mean, var = beta.stats(b_dist_a,b_dist_b,moments='mv')
			# assume rmag and lmag are same for both options
			infer_val = self.r_mag[0]*mean + self.l_mag[0]*(1-mean)
			PE = reward - infer_val 
		else:
			err = 'Invalid value given for arg crit. %s given' %crit
			raise Exception(err)
		self.PE[idx] = PE


	def policy_forced (self,C):
		""" Forced action selection """
		idx = self.idx 				# internal time index of state
		probs = self.probs			# prob of reward for an action
		crit  = self.crit    		# critic type

		self.C[idx] = C 			# forced option choice

		# NOTE: does not update SM for chosen option		

		# decide whether a reward is delivered
		reward = np.random.binomial(size=1, n=1, p= probs[C])[0]
		self.R[idx] = reward 		# indicator that reward was received
		if reward == 0:
			reward = self.l_mag[C]
		else:
			reward = self.r_mag[C]
		
		# calculate PE
		if crit == "S":
			PE = reward - self.V[idx]
		elif crit == "SA":
			PE = reward - self.V[idx,C]
		elif crit == "Bayes":
			b_dist = self.V[idx,:]
			mean, var = beta.stats(b_dist[0],b_dist[1],moments='mv')
			# assume rmag and lmag are same for both options
			infer_val = self.r_mag[0]*mean + self.l_mag[0]*(1-mean)
			PE = reward - infer_val 
		elif crit == "Bayes-SA":
			choice_idx = C*2
			b_dist_a = self.V[idx,choice_idx]
			b_dist_b = self.V[idx,choice_idx+1]
			mean, var = beta.stats(b_dist_a,b_dist_b,moments='mv')
			# assume rmag and lmag are same for both options
			infer_val = self.r_mag[0]*mean + self.l_mag[0]*(1-mean)
			PE = reward - infer_val  
		else:
			err = 'Invalid value given for arg crit. %s given' %crit
			raise Exception(err)
		self.PE[idx] = PE

	def policy_gamble (self,thresh=0):
		"""
		Decides whether or not to take a gamble over a default option
		"""
		idx = self.idx 				# internal time index of state
		probs = self.probs			# prob of reward for an action
		beta_g = self.beta_g[idx]	# inverse temp for go action value	
		beta_n = self.beta_n[idx]	# inverse temp for nogo action value
		crit  = self.crit    		# critic type
		C = self.C[idx] 			# forced option choice

		# softmax policy
		if idx < thresh:
			# random policy
			p = 0.5
		else:
			Act = beta_g*self.QG[idx] - beta_n*self.QN[idx]
			self.Act[idx] = Act
			p = 1./(1. + np.exp(-Act))	# probability of gamble
		self.SM[idx] = p

		# decide whether to take gamble based on p
		rnd = np.random.random_sample()
		if rnd < p:
			C = 1	# gamble
		else:
			C = 0	# no gamble
		self.C[idx] = C

		# no gamble
		if C == 0:	
			reward = 0		  # gamble reward encoded relative to reward
			self.R[idx] = -1  # rewarded sure thing, coded as -1
			self.PE[idx] = 0  # no PE, get the thing you expected
		# gamble
		else:
			# decide whether a reward is delivered
			reward = np.random.binomial(size=1, n=1, p=probs)[0]
			self.R[idx] = reward # indicator that reward was received
			if reward == 0:
				if self.env[0:2] == "EC":
					reward = self.l_mag[idx]
				else:
					reward = self.l_mag[0]
			else:
				if self.env[0:2] == "EC":
					reward = self.l_mag[idx]
				else:
					reward = self.r_mag[0]
			
			PE = None
			if crit == "S":
				PE = reward - self.V[idx]
			elif crit == "Bayes":
				b_dist = self.V[idx,:]
				mean, var = beta.stats(b_dist[0],b_dist[1],moments='mv')
				# assume rmag and lmag are same for both options
				infer_val = self.r_mag*mean + self.l_mag*(1-mean)
				PE = reward - infer_val 
			elif crit == "Bayes-SA":
				b_dist_a = self.V[idx,0]
				b_dist_b = self.V[idx,1]
				mean, var = beta.stats(b_dist_a,b_dist_b,moments='mv')
				# assume rmag and lmag are same for both options
				infer_val = self.r_mag[0]*mean + self.l_mag[0]*(1-mean)
				PE = reward - infer_val 
			self.PE[idx] = PE

# Tracks and updates information for each trial across ALL STATES/OpAL instances
# Used when agent is experiencing learning across multiple states that are
# interleaved, which may generate state PEs
class TrialTracker:
	def __init__(self, n_states):
		self.idx = 0                                # which state to update
		self.state_priors = np.empty((n_states,2)) 	# current state priors
		self.state_priors[:] = np.nan	



	#### Other draft act variants ####
	# def act_normalize_PE (self,alpha,mag=1):
	# 	"""
	# 	Normalizes PE by "memory" of largest recent
	# 	magnitude reward
	# 	"""
	# 	idx = self.idx
	# 	PE = self.PE[idx]
	# 	C = self.C[idx]	# choice

	# 	self.QG[idx+1] = self.QG[idx] #carry over values
	# 	self.QN[idx+1] = self.QN[idx] 

	# 	# normalize PE between -1 and 1 relative to largest
	# 	# experienced reward magnitude
	# 	normPE = np.tanh(PE/mag)
	# 	#normPE = 2*(1/(1 + np.exp(-PE/mag))-.5)

	# 	# scale update speed by floating threshold
	# 	self.QG[idx+1,C] = self.QG[idx,C] + alpha*normPE*self.QG[idx,C]
	# 	self.QN[idx+1,C] = self.QN[idx,C] + alpha*-normPE*self.QN[idx,C]

	# 	# actor values should not be negative, represent population activations
	# 	self.QG[idx+1,C] = np.max([0,self.QG[idx+1,C]])
	# 	self.QN[idx+1,C] = np.max([0,self.QN[idx+1,C]])

	# def act_normalize_PE_prob (self,alpha,mag=1):
	# 	"""
	# 	Normalizes PE by "memory" of largest recent
	# 	magnitude reward, update QG according to probability
	# 	of selecting that option
	# 	"""
	# 	idx = self.idx
	# 	PE = self.PE[idx]
	# 	C = self.C[idx]	# choice

	# 	self.QG[idx+1] = self.QG[idx] #carry over values
	# 	self.QN[idx+1] = self.QN[idx] 

	# 	# normalize PE between -1 and 1 relative to largest
	# 	# experienced reward magnitude
	# 	normPE = np.tanh(PE/mag)

	# 	# determine strength of update by probability of chosing action
	# 	update = 1 - self.SM[idx,C]

	# 	# scale update speed by floating threshold
	# 	self.QG[idx+1,C] = self.QG[idx,C] + alpha*normPE*self.QG[idx,C]*update
	# 	self.QN[idx+1,C] = self.QN[idx,C] + alpha*-normPE*self.QN[idx,C]*update

	# 	# actor values should not be negative, represent population activations
	# 	self.QG[idx+1,C] = np.max([0,self.QG[idx+1,C]])
	# 	self.QN[idx+1,C] = np.max([0,self.QN[idx+1,C]])

	# def act_normalize_PE_avg (self,alpha,mag=1,beta=1):
	# 	"""
	# 	Normalizes PE by "memory" of largest recent
	# 	magnitude reward
	# 	"""
	# 	idx = self.idx
	# 	PE = self.PE[idx]
	# 	C = self.C[idx]	# choice

	# 	self.QG[idx+1] = self.QG[idx] #carry over values
	# 	self.QN[idx+1] = self.QN[idx] 

	# 	# normalize PE between -1 and 1 relative to largest
	# 	# experienced reward magnitude
	# 	normPE = np.tanh(PE/mag)

	# 	# determine strength of nonlinearity
	# 	gbar = np.mean(self.QG[idx])
	# 	nbar = np.mean(self.QN[idx])
	# 	logitG = 1/(1 + np.exp(beta*(self.QG[idx,C] - gbar)))
	# 	logitN = 1/(1 + np.exp(beta*(self.QN[idx,C] - nbar)))

	# 	# scale update speed by floating threshold
	# 	self.QG[idx+1,C] = self.QG[idx,C] + alpha*logitG*self.QG[idx,C]*normPE
	# 	self.QN[idx+1,C] = self.QN[idx,C] + alpha*logitN*self.QN[idx,C]*-normPE

	# 	# actor values should not be negative, represent population activations
	# 	self.QG[idx+1,C] = np.max([0,self.QG[idx+1,C]])
	# 	self.QN[idx+1,C] = np.max([0,self.QN[idx+1,C]])

	# def act_normalize_PE_all (self,alpha,mag=1,beta=1):
	# 	"""
	# 	Normalizes PE by "memory" of largest recent
	# 	magnitude reward
	# 	"""
	# 	idx = self.idx
	# 	PE = self.PE[idx]
	# 	C = self.C[idx]	# choice

	# 	self.QG[idx+1] = self.QG[idx] #carry over values
	# 	self.QN[idx+1] = self.QN[idx] 

	# 	# normalize PE between -1 and 1 relative to largest
	# 	# experienced reward magnitude
	# 	normPE = np.tanh(PE/mag)

	# 	# determine strength of nonlinearity
	# 	gbar = np.mean(self.QG[idx])
	# 	nbar = np.mean(self.QN[idx])
	# 	thresh = (gbar + nbar)/2.
	# 	logitG = 1/(1 + np.exp(beta*(self.QG[idx,C] - thresh)))
	# 	logitN = 1/(1 + np.exp(beta*(self.QN[idx,C] - thresh)))

	# 	# scale update speed by floating threshold
	# 	self.QG[idx+1,C] = self.QG[idx,C] + alpha*normPE*logitG*self.QG[idx,C]
	# 	self.QN[idx+1,C] = self.QN[idx,C] + alpha*-normPE*logitN*self.QN[idx,C]

	# 	# actor values should not be negative, represent population activations
	# 	self.QG[idx+1,C] = np.max([0,self.QG[idx+1,C]])
	# 	self.QN[idx+1,C] = np.max([0,self.QN[idx+1,C]])

	# def act_normalize_avg (self,alpha,beta=1):
	# 	"""
	# 	Normalizes hebbian update by average population firing
	# 	in respective G and N populations
	# 	"""
	# 	idx = self.idx
	# 	PE = self.PE[idx]
	# 	C = self.C[idx]	# choice

	# 	self.QG[idx+1] = self.QG[idx] #carry over values
	# 	self.QN[idx+1] = self.QN[idx] 

	# 	# determine strength of nonlinearity
	# 	gbar = np.mean(self.QG[idx])
	# 	nbar = np.mean(self.QN[idx])
	# 	logitG = 1/(1 + np.exp(beta*(self.QG[idx,C] - gbar)))
	# 	logitN = 1/(1 + np.exp(beta*(self.QN[idx,C] - nbar)))

	# 	# scale update speed by floating threshold
	# 	self.QG[idx+1,C] = self.QG[idx,C] + alpha*logitG*self.QG[idx,C]*PE
	# 	self.QN[idx+1,C] = self.QN[idx,C] + alpha*logitN*self.QN[idx,C]*-PE

	# 	# actor values should not be negative, represent population activations
	# 	self.QG[idx+1,C] = np.max([0,self.QG[idx+1,C]])
	# 	self.QN[idx+1,C] = np.max([0,self.QN[idx+1,C]])

	# def act_normalize_all (self,alpha,beta=1):
	# 	"""
	# 	Normalizes hebbian update by average firing across
	# 	average G/N population level
	# 	"""
	# 	idx = self.idx
	# 	PE = self.PE[idx]
	# 	C = self.C[idx]	# choice

	# 	self.QG[idx+1] = self.QG[idx] #carry over values
	# 	self.QN[idx+1] = self.QN[idx] 

	# 	# determine strength of nonlinearity
	# 	gbar = np.mean(self.QG[idx])
	# 	nbar = np.mean(self.QN[idx])
	# 	thresh = (gbar + nbar)/2.
	# 	logitG = 1/(1 + np.exp(beta*(self.QG[idx,C] - thresh)))
	# 	logitN = 1/(1 + np.exp(beta*(self.QN[idx,C] - thresh)))

	# 	# scale update speed by floating threshold
	# 	self.QG[idx+1,C] = self.QG[idx,C] + alpha*logitG*self.QG[idx,C]*PE
	# 	self.QN[idx+1,C] = self.QN[idx,C] + alpha*logitN*self.QN[idx,C]*-PE

	# 	# actor values should not be negative, represent population activations
	# 	self.QG[idx+1,C] = np.max([0,self.QG[idx+1,C]])
	# 	self.QN[idx+1,C] = np.max([0,self.QN[idx+1,C]])

	# def act_counterfact (self,alpha):
	# 	"""   G for the chosen action goes up proportional to the 
	# 	sum of G's across all other actions, and then a counterfactual update
	# 	"""
	# 	pass

	# def act_prob (self,alpha):
	# 	""" Probability of choosing more, 
	# 	value updates less
	# 	"""
	# 	idx = self.idx
	# 	PE = self.PE[idx]
	# 	C = self.C[idx]	# choice

	# 	self.QG[idx+1] = self.QG[idx] #carry over values
	# 	self.QN[idx+1] = self.QN[idx] 

	# 	# determine strength of update by probability of chosing action
	# 	update = 1 - self.SM[idx,C]

	# 	# scale update speed by floating threshold
	# 	self.QG[idx+1,C] = self.QG[idx,C] + alpha*update*self.QG[idx,C]*PE
	# 	self.QN[idx+1,C] = self.QN[idx,C] + alpha*update*self.QN[idx,C]*-PE

	# 	# actor values should not be negative, represent population activations
	# 	self.QG[idx+1,C] = np.max([0,self.QG[idx+1,C]])
	# 	self.QN[idx+1,C] = np.max([0,self.QN[idx+1,C]])

	# def act_softbound (self,alpha,maxq):
	# 	"""
	# 	Updates the Q vales for the direct (G) and indirect (NG) actors
	# 	for the next time step for chosen action with a soft upper bound 
	# 	on the weights
	# 	"""
	# 	idx = self.idx
	# 	PE = self.PE[idx]
	# 	C = self.C[idx]	# choice

	# 	self.QG[idx+1] = self.QG[idx] #carry over values
	# 	self.QN[idx+1] = self.QN[idx] 

	# 	# update chosen with soft upper bound
	# 	self.QG[idx+1,C] = self.QG[idx,C] + (maxq - self.QG[idx,C])*alpha*self.QG[idx,C]*PE
	# 	self.QN[idx+1,C] = self.QN[idx,C] + (maxq - self.QN[idx,C])*alpha*self.QN[idx,C]*-PE

	# 	# actor values should not be negative, represent population activations
	# 	self.QG[idx+1,C] = np.max([0,self.QG[idx+1,C]])
	# 	self.QN[idx+1,C] = np.max([0,self.QN[idx+1,C]])
