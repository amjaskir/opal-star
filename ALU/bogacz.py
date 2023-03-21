# Lise VANSTEEENKISTE
# internship

# Edits by Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank

# Name: bogacz.py


import numpy as np
import random

class Bogacz:
    def __init__(self, n_trials, n_options, probs, r_mag, l_mag, V0=0.0):

        sz = n_trials

        self.n_options = n_options
        self.QG = np.zeros((sz + 1, n_options)) + V0  # Go Actor Values
        self.QN = np.zeros((sz + 1, n_options)) + V0  # NoGo Actor Values
        self.rho = np.zeros(sz)
        self.D_g = np.zeros(sz) + 1/2  # dopamine level at choice, contribution of go
        self.D_n = np.zeros(sz) + 1/2  # 1- D_g
        self.SM = np.zeros((sz, n_options))  # Softmax values
        self.C = np.zeros(sz, dtype=np.int)  # choice
        self.R = np.zeros(sz, dtype=np.int)  # indicator variable of reward
        self.probs = probs  # prob of reward for choice
        self.r_mag = r_mag  # magnitude of reward for choice
        self.l_mag = l_mag  # magnitude of loss for choice
        self.random_noise = np.zeros(sz)
        self.PE = np.zeros(sz)  # choice PE ( delta )
        self.idx = 0  # idx of trial in state
        self.prior = np.zeros((sz + 1, 2)) + .5  # probability high or low state


    def policy_max(self, K):
        """ Select option with max thalamic output, given some
        random noise is added independently to each output.
        K gives range of noise, equiv of explore/exploit temp
        """

        idx = self.idx      # internal time index of state
        probs = self.probs  # prob of reward for an action
        D_g = self.D_g[idx]
        D_n = self.D_n[idx]

        random_noiseA = (np.random.random_sample()-.5)*K   # explore with the value K
        random_noiseB = -(np.random.random_sample() - .5)*K  # explore with the value K
        Act = D_g*self.QG[idx]-D_n*self.QN[idx]

        Act_a = Act[0] + random_noiseA
        Act_b = Act[1] + random_noiseB
        C = np.where([Act_a, Act_b] == np.max([Act_a, Act_b]))[0][0]
        self.C[idx] = C

        # decide whether a reward is delivered
        reward = np.random.binomial(size=1, n=1, p=probs[C])[0]

        self.R[idx] = reward  # indicator that reward was received
        if reward == 0:
            reward = self.l_mag[C]
        else:
            reward = self.r_mag[C]

        # calculate PE
        PE = reward - 1/2*(self.QG[idx, C]-self.QN[idx, C])
        self.PE[idx] = PE

    def policy_softmax(self, beta):
        """
        Selects one of two actions via softmax.
        Activation function is a linear combination of the two actors
        according to betaG/betaN, asymmetry determined by tonic DA
        """
        idx = self.idx  # internal time index of state
        probs = self.probs  # prob of reward for an action
        D_g = self.D_g[idx]
        D_n = self.D_n[idx]

        # softmax policy
        Act = beta*(D_g*self.QG[idx]-D_n*self.QN[idx])

        # multioption softmax (invariant to constant offsets)
        newAct = Act - np.max(Act)
        expAct = np.exp(newAct)
        ps = expAct/np.sum(expAct)
        self.SM[idx,:] = ps
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
        self.C[idx] = C

        # decide whether a reward is delivered
        reward = np.random.binomial(size=1, n=1, p=probs[C])[0]
        self.R[idx] = reward  # indicator that reward was received
        if reward == 0:
            reward = self.l_mag[C]
        else:
            reward = self.r_mag[C]

        # calculate PE
        PE = reward - 1/2*(self.QG[idx, C]-self.QN[idx, C])   #should this also be affected by D, 2016 no 1/2
        self.PE[idx] = PE

    def compute_pe(self, epsilon, PE):
        """
        function epsilon the strength of the nonlinearity 
        exhibited by the function fepslion.
        """
        if PE < 0:
            return epsilon*PE
        else:
            return PE


    def act(self, alpha, epsilon, lbda):

        idx = self.idx
        PE = self.PE[idx]
        C = self.C[idx]  # choice

        self.QG[idx + 1] = self.QG[idx]
        self.QN[idx + 1] = self.QN[idx]

        self.QG[idx + 1, C] = np.max([0,self.QG[idx, C] + alpha*self.compute_pe(epsilon, PE)-lbda*self.QG[idx, C]])
        self.QN[idx + 1, C] = np.max([0,self.QN[idx, C] + alpha*self.compute_pe(epsilon, -PE)-lbda*self.QN[idx, C]])

    def act_simple(self, alpha, epsilon, lbda):

        idx = self.idx
        PE = self.PE[idx]
        C = self.C[idx]  # choice

        self.QG[idx + 1] = self.QG[idx]
        self.QN[idx + 1] = self.QN[idx]

        self.QG[idx + 1, C] = np.max([0,self.QG[idx, C] + alpha*PE])
        self.QN[idx + 1, C] = np.max([0,self.QN[idx, C] + alpha*-PE])


    def update_other_actions(self, alpha_a, epsilon, lbda):
        """
        Cycle through all unselected actions and force model to select them.
        Call after standard policy()/actor() functions are called
        with same arguments as they were provided
        """
        update = np.arange(self.n_options)
        update = np.delete(update,self.C) # remove chosen option from update

        for c in update:
            self.policy_forced(c)
            self.act(alpha_a, epsilon, lbda)
            

    ######## Variants #########
    def policy_forced(self, C):
        """ Forced action selection """
        idx = self.idx  # internal time index of state
        probs = self.probs  # prob of reward for an action

        self.C[idx] = C  # forced option choice

        # decide whether a reward is delivered
        reward = np.random.binomial(size=1, n=1, p=probs[C])[0]
        self.R[idx] = reward  # indicator that reward was received
        if reward == 0:
            reward = self.l_mag[C]
        if reward == 1:
            reward = self.r_mag[C]

        # calculate PE
        PE = reward - 1/2*(self.QG[idx, C]-self.QN[idx, C])

        self.PE[idx] = PE

    def policy_gamblesoftmax(self):
        """
        Decides whether or not to take a gamble over a default option
        """
        idx = self.idx  # internal time index of state
        probs = self.probs  # prob of reward for an action
        D_g = self.D_g[idx]  # inverse temp for go action value
        D_n = self.D_n[idx]  # inverse temp for nogo action value
        C = self.C[idx]  # forced option choice

        # softmax policy
        Act = D_g * self.QG[idx] - D_n * self.QN[idx]
        p = 1. / (1. + np.exp(-Act))  # probability of gamble
        self.SM[idx] = p

        # decide whether to take gamble based on p
        rnd = np.random.random_sample()
        if rnd < p:
            C = 1
        else:
            C = 0
        self.C[idx] = C

        # no gamble
        if C == 0:
            reward = 0  # gamble reward encoded relative to reward			#### unclear if this PE should be 0 (original interpretation)
            self.R[idx] = 1
        # gamble
        else:
            # decide whether a reward is delivered
            reward = np.random.binomial(size=1, n=1, p=probs)[0]
            self.R[idx] = reward  # indicator that reward was received
            if reward == 0:
                reward = self.l_mag
            else:
                reward = self.r_mag


        self.PE[idx] = reward - 1/2*(self.QG[idx]-self.QN[idx])

    def act_gamble(self, alpha, epsilon, lbda):

        idx = self.idx
        PE = self.PE[idx]
        C = self.C[idx]  # choice

        self.QG[idx + 1] = self.QG[idx]  # carry over values
        self.QN[idx + 1] = self.QN[idx]

        # took the gamble
        if C == 1:

            self.QG[idx + 1] = self.QG[idx] + alpha * self.compute_pe(epsilon, PE) - lbda * self.QG[idx]
            self.QN[idx + 1] = self.QN[idx] + alpha * self.compute_pe(epsilon, -PE) - lbda * self.QN[idx]


class TrialTracker:
    def __init__(self, tot_trials, n_states):
        sz = tot_trials
        self.state_idx = []  # 0 - S, where S is the number of states
        self.PE = np.zeros(sz + 1)  # trial-level PE

        ### Simulation specific elements ###
        self.state_vs = np.empty(n_states)  # state values, initialzied to NaN til encountered
        self.state_vs[:] = np.nan
        self.state_priors = np.empty((n_states, 2))  # current state priors
        self.state_priors[:] = np.nan