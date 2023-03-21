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

class RL:
	def __init__(self, n_trials, v0, n_options, probs, r_mag, l_mag, params, gamble, \
		UCB=False, UCB_samplemean=False, gamma=0, anneal=False):
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

		sz = n_trials
		self.n_options = n_options					# number of choices
		self.Q = np.zeros((sz+1,n_options)) +	v0	# critic
		self.alpha, self.beta = params				# learning rate, softmax
		self.SM  = np.zeros((sz, n_options))		# Softmax values
		self.rho = np.zeros(sz)						# DA at choice
		self.C = np.zeros(sz,dtype=np.int)			# choice
		self.R = np.zeros(sz,dtype=np.int)			# indicator variable of reward
		self.probs = probs							# prob of reward for choice
		self.r_mag = r_mag							# magnitude of reward for choice
		self.l_mag = l_mag							# magnitude of loss for choice
		self.PE  = np.zeros(sz)						# choice PE
		self.idx = 0								# idx of trial in state  
		self.gamble = gamble						# gamble paradigm?
		self.UCB = UCB								# Use UCB for policy?	
		self.UCB_samplemean = UCB_samplemean		# Use sample mean instead of Q value
		self.gamma = gamma							# Exploration parameter	
		self.N = np.zeros((sz+1, n_options))		# number of times action selected
		self.R_by_a = np.zeros((sz+1, n_options))	# number of times rewarded for action
		self.Acts = np.zeros((sz+1,n_options))		# act values

	def policy_UCB (self,forced_actions=None,forced_rewards=None,state_idx=None):
		idx = self.idx 				# internal time index of state
		probs = self.probs			# prob of reward for an action
		Qs = self.Q[idx,:]			# Q-values for given trial
		Ns = self.N[idx,:]			# Action selection counts
		gamma = self.gamma			# uncertainty bonus hyperparameter

		unselected = np.where(self.N[idx,:] == 0)[0]
		if len(unselected) > 0: 
			# if any actions haven't been chosen, choose between them randomly
			C = np.random.choice(unselected)
			self.Acts[idx,unselected] = 1	# maximal Q for unselected 
			self.SM[idx,unselected] = 1/(len(unselected)) # update SM, though technically no SM and deterministic

		else:

			# use sample mean instead of q?
			if self.UCB_samplemean:
				Qs = self.R_by_a[idx,:]/Ns
				self.Q[idx,:] = Qs

			# add uncertainty bonus to exploitation value
			uncertainty_bonus = gamma*np.sqrt(np.log(idx+1))/np.sqrt(Ns) # use +1 to avoid nan
			Act = Qs + uncertainty_bonus
			self.Acts[idx,:] = Act  # store

			# just take the max
			C = np.random.choice(np.where(Act == Act.max())[0])
			self.SM[idx,C] = 1 # update SM, though technically no SM and deterministic

		if forced_actions is not None:
			C = forced_actions[state_idx,idx]

		# save choice
		self.C[idx]= C

		# update choice counts for next trial
		self.N[idx+1,:] = self.N[idx,:]
		self.N[idx+1,C] = self.N[idx+1,C] + 1
		self.R_by_a[idx+1,:] = self.R_by_a[idx,:]

		# decide whether a reward is delivered
		if forced_rewards is not None:
			reward = forced_rewards[state_idx,idx]
		else:
			reward = np.random.binomial(size=1, n=1, p= probs[C])[0]
		self.R[idx] = reward # indicator that reward was received
		self.R_by_a[idx+1,C] = self.R_by_a[idx+1,C] + reward # count of reward per action
		if reward == 0:
			reward = self.l_mag
		else:
			reward = self.r_mag

		PE = reward - self.Q[idx,C]
		self.PE[idx] = PE


	def policy (self,forced_actions=None,forced_rewards=None,state_idx=None):
		"""
		Selects an action via softmax. 
		Activation function is a linear combination of the two actors 
		according to betaG/betaN, asymmetry determined by tonic DA
		"""
		if self.gamble:
			self.policy_gamble()
			return
		if self.UCB:
			self.policy_UCB(forced_actions,forced_rewards,state_idx)
			return

		idx = self.idx 				# internal time index of state
		probs = self.probs			# prob of reward for an action
		beta = self.beta			# inverse temp 

		# calc Act thalamus activation
		Act = beta*self.Q[idx,:]

		# multioption softmax (invariant to constant offsets)
		newAct = Act - np.max(Act)
		expAct = np.exp(newAct)
		ps = expAct/np.sum(expAct)
		self.SM[idx,:] = ps
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
			
		# decide whether a reward is delivered
		if forced_rewards is None:
			reward = np.random.binomial(size=1, n=1, p= probs[C])[0]
		else:
			reward = forced_rewards[state_idx,idx]
		self.R[idx] = reward # indicator that reward was received
		if reward == 0:
			reward = self.l_mag
		else:
			reward = self.r_mag

		PE = reward - self.Q[idx,C]
		self.PE[idx] = PE

	def policy_gamble (self):
		"""
		Decides whether or not to take a gamble over a default option
		"""
		idx = self.idx 				# internal time index of state
		probs = self.probs			# prob of reward for an action
		beta = self.beta			# inverse temp 

		# softmax
		Act = beta*self.Q[idx]
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
				reward = self.l_mag
			else:
				reward = self.r_mag
			self.PE[idx] = reward - self.Q[idx]

	def update (self):
		""" Updates value for chosen action/gamble """
		idx = self.idx
		C = self.C[idx]		# choice
		PE = self.PE[idx]	# choice PE
		alpha = self.alpha	# learning rate

		# don't need to update anything for UCB
		if self.UCB_samplemean:
			return

		if not self.gamble:
			# carry over values for the unselected options
			self.Q[idx+1,:] = self.Q[idx,:]
			# check if two learning rates (pos/neg)
			if isinstance(alpha,float):
				self.Q[idx+1,C] = self.Q[idx,C] + alpha*PE
			else:
				if PE > 0:
					self.Q[idx+1,C] = self.Q[idx,C] + alpha[0]*PE
				else:
					self.Q[idx+1,C] = self.Q[idx,C] + alpha[1]*PE

		else:
			# check if two learning rates (pos/neg)
			# PE = 0 if gamble isn't chosen
			if isinstance(alpha,float):
				self.Q[idx+1] = self.Q[idx] + alpha*PE
			else:
				if PE > 0:
					self.Q[idx+1] = self.Q[idx] + alpha[0]*PE
				else:
					self.Q[idx+1] = self.Q[idx] + alpha[1]*PE



