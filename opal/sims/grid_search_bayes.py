##################################################################
# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
#
# Description: searches over outlined parameter space for learning.py
# Splits search into parameter sweep chunks
# Saves states, AUC and curves for learning and reward
# OpAL model uses a beta critic  
#
# Name: opal/grid_search_bayes.py
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


def main(job,env,n_trials,n_states,norm,split):
    """ 
    Run specified parameter combinations for desired environment.
    Saves data to specified environment

    Inputs:
    job         - idx of parameter combination to try
    env         - specified by environments.py
    n_trials    - number of trials for learning
    n_states    - number of states to be averaging accuracy over
    split       - number of splits for parameter combinations
    """

    t = time.time()
    these_params = params.get_params(job,split,"bayes")
    ks = np.round(np.arange(0,10.1),1) # k=0 means no modulation

    for group in these_params:

        # new random seed for each parameter combination
        # each modulation level shares same random seed
        rnd_seed = np.random.randint(1,100000)
        for k in ks:

            print(group)
            # get args
            pars = tuple(group[0:3])
            print(pars)
            lmbda = group[3]
            v0 = 0.0
            crit = "Bayes"

            # get appropriate mod arg
            if k == 0:
                mod = "constant"
            else:
                mod = "beta"

            # learn the things
            states = learning.simulate(pars,n_states,n_trials,v0=v0,crit=crit,\
                env=env,mod=mod,k=k,lmbda=lmbda,norm=norm,rnd_seed=rnd_seed)

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
            path = './results/grid_bayes/%s/norm_%s/k_%s/l_%s/mod_%s/' %(env,str(norm),k,lmbda,mod)
            os.makedirs(path, exist_ok=True)  #create directory when non-existing
            pickle.dump([auc_learn, auc_reward, learn_curve, reward_curve], \
                open("results/grid_bayes/%s/norm_%s/k_%s/l_%s/mod_%s/params_%s.pkle" \
                %(env,str(norm),k,lmbda,mod,pars[1:]),"wb"))

    elapsed = time.time() - t
    print('Time elapsed: %f' % (elapsed))
    sys.stdout.flush()


if __name__ == '__main__':
    main(int(sys.argv[1]),sys.argv[2],int(sys.argv[3]),\
        int(sys.argv[4]),bool(int(sys.argv[5])),int(sys.argv[6]))
