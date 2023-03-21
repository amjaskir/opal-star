##################################################################
# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
#
# Description: searches over outlined parameter space for learning.py
# Splits search into parameter sweep chunks
# Saves states, AUC and curves for learning and reward in varying 
# levels of domain complexity
# OpAL model uses a beta critic  
#
# Name: opal/probability_space.py
##################################################################

import itertools
import pickle
import scipy.stats as stats
import numpy as np
from sklearn import metrics
import time

# get my helper modules
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../../helpers'))
import params
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import learning


def main(job,n_trials,n_states,split,opt_prob):
    """ 
    Run specified parameter combinations for desired environment.
    Saves data to specified environment

    Inputs:
    job         - idx of parameter combination to try
    n_trials    - number of trials for learning
    n_states    - number of states to be averaging accuracy over
    split       - number of splits for parameter combinations
    opt_prob    - 20 ro 90 % prob of reward
    """
    t = time.time()  # each level is about 30 min for 1 lambda
    root_me = "complexity"

    #######################################################
    # Various settings - restriction, full, anneal
    # specify save destination

    # give model full info?
    full = False
    # TODO: need to change save path if this is true

    # anneal learning rate?
    anneal = True
    T = 100.0
    if anneal:
        base = "anneal_%d/" %T
        root_me = base + root_me

    # constant parameters for learning
    v0 = 0.0
    crit = "Bayes"
    norm = True
    ks = np.array([0,20]) #0 is no modulation

    # DA modulation variations
    # mod_var = "beta"
    mod_var = "beta_50_phi3"
    par_key = "extrange"
    #######################################################
    these_params = params.get_params(job,split,par_key)
    n_levels = 8
    levels = np.arange(n_levels) + 2 # number of levels of complexity
    step_sz = 10    #step size in %

    # compute AUC for each level
    for level in levels:

        # get environment name
        env = "%s_%d_%d" %(opt_prob,step_sz,level)

        # do the thing
        for group in these_params:

            # check if I already have these params

            # new random seed for each parameter combination
            # each modulation level shares same random seed
            rnd_seed = np.random.randint(1,100000)
            for k in ks:

                # get args
                pars = tuple(group[0:3])
                lmbda = group[3]

                # get appropriate mod arg
                if k == 0:
                    mod = "constant"
                else:
                    mod = mod_var

                # learn the things
                states = learning.simulate(pars,n_states,n_trials,v0=v0,crit=crit,\
                    env=env,mod=mod,k=k,lmbda=lmbda,norm=norm,rnd_seed=rnd_seed,\
                    anneal=anneal,T=T)

                # calc learning curve according to softmax 
                # calc reward curve 
                # first option is always best option
                first = True
                for state in states:
                    if first:
                        sms = state.SM[:,0]
                        rs = state.R
                        first = False
                    else:
                        sms = np.vstack([state.SM[:,0],sms])
                        rs = np.vstack([state.R,rs])
                avg_sm = np.mean(sms, axis = 0)
                sem_sm = stats.sem(sms, axis = 0)
                avg_r = np.mean(rs, axis = 0)
                sem_r = stats.sem(rs, axis = 0)

                learn_curve = (avg_sm,sem_sm)
                reward_curve = (avg_r,sem_r)

                # calc AUC of both
                x = np.arange(n_trials)
                auc_learn = metrics.auc(x,avg_sm)
                auc_reward = metrics.auc(x,avg_r)

                # fix extentions depending on version
                if k == 0:
                    mod = "constant_" + mod_var
                par_str = "%s" %(str(pars[1:]))

                # save all that work in env specific folder
                path = './results/%s_%d_%d/%s/k_%s/mod_%s/' \
                        %(root_me,n_trials,n_states,env,k,mod)
                os.makedirs(path, exist_ok=True)  #create directory when non-existing
                pickle.dump([auc_learn, auc_reward, learn_curve, reward_curve, rnd_seed], \
                        open("results/%s_%d_%d/%s/k_%s/mod_%s/params_%s.pkle" \
                        %(root_me,n_trials,n_states,env,k,mod,par_str),"wb"))


        # speed for all the levels together
        elapsed = time.time() - t
        print('level %d done' % level)
        print('Time elapsed: %f' % (elapsed))
        sys.stdout.flush()


    # speed for all the levels together
    elapsed = time.time() - t
    print('Time elapsed: %f' % (elapsed))
    sys.stdout.flush()


if __name__ == '__main__':
    main(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),\
        int(sys.argv[4]),sys.argv[5])
