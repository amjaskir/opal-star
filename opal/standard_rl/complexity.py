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
# Name: standard_rl/grid_search.py
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
import learn


def main(job,env_base,n_trials,n_states,split):
    """ 
    Run specified parameter combinations for desired environment.
    Saves data to specified environment

    Inputs:
    job         - idx of parameter combination to try
    env_base    - "rich", "richmix", "lean", "leanmix", "mix"
    n_trials    - number of trials for learning
    n_states    - number of states to be averaging accuracy over
    split       - number of splits for parameter combinations
    variant     - variant of opal to be run
                  "flip", "bmod","lrate"
    """
    t = time.time()  # each level is about 30 min
    par_key = "extrange"
    these_params = params.get_params(job,split,par_key)
    ks = params.get_ks(par_key)
    n_levels = 8 # number of levels of complexity

    # specify save destination
    root_me = "complexity"
    v0 = 0.0

    # compute AUC for each level
    for level in np.arange(n_levels):

        # get environment name
        if level == 0:
            n_opt = ""                    # no number for base environment
        else:   
            n_opt = "_" + str(level + 2)  # offset by 2
        env = env_base + n_opt

        # do the thing
        for group in these_params:
            # new random seed for each parameter combination
            # each modulation level shares same random seed
            rnd_seed = np.random.randint(1,100000)

            # get argsm alpha/beta
            pars = tuple(group[1:3])

            # learn the things
            states = learn.simulate(pars,n_states,n_trials,v0=v0,
                env=env,r_mag=1,l_mag=-1,rnd_seed=rnd_seed)

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

            # save all that work in env specific folder
            path = './results/%s_%d_%d/%s/%s/' \
                    %(root_me,n_trials,n_states,env_base,env)
            os.makedirs(path, exist_ok=True)  #create directory when non-existing
            pickle.dump([auc_learn, auc_reward, learn_curve, reward_curve, rnd_seed], \
                    open("results/%s_%d_%d/%s/%s/params_%s.pkle" \
                    %(root_me,n_trials,n_states,env_base,env,pars),"wb"))


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
    main(int(sys.argv[1]),sys.argv[2],int(sys.argv[3]),\
        int(sys.argv[4]),int(sys.argv[5]))
