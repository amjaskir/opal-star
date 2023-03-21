# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank

# Name: grid_search.py
# Description: searches over outlined parameter space for learning.py
# Splits search into parameter sweep chunks
# Saves states, AUC and curves for learning and reward

import itertools
import pickle
import scipy.stats as stats
import numpy as np
from sklearn import metrics
import time

# get my helper modules
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import learning

def get_params(idx,split):
    """ get the list of params for this job idx 
        divides combination list into split # of groups 
        n_total combinations = 800
    """

    # param combos
    alpha_cs = np.round(np.arange(.1,1.1,.1),1)
    alpha_as = np.round(np.arange(.1,1.1,.1),1)
    betas = np.round(np.arange(.25,2.1,.25),2)
    all_combos = list(itertools.product(*[alpha_cs,alpha_as,betas]))
    divide_n_conquer = np.array_split(all_combos, split)

    return divide_n_conquer[idx]


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
    these_params = get_params(job,split)
    ks = np.round(np.arange(0,10.1),1) # k=0 means no modulation

    for group in these_params:

        # new random seed for each parameter combination
        # each modulation level shares same random seed
        rnd_seed = np.random.randint(1,100000)
        for k in ks:

            # get args
            params = tuple(group[0:3])
            v0 = 0.0
            crit = "S"

            # only run if original params
            # to debug
            # if params[0] != .1 or params[1] != .3 or params[2] != 1.5 or (k not in [5.,0]):
            #     continue
            # rnd_seed = 23023
            # print(rnd_seed, sys.version_info)

            # get appropriate mod arg
            if k == 0:
                mod = "constant"
            else:
                mod = "value"

            # learn the things
            states = learning.simulate(params,n_states,n_trials,v0=v0,crit=crit,\
                env=env,mod=mod,k=k,norm=norm, rnd_seed = rnd_seed)

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
            path = './results/grid/%s/norm_%s/k_%s/mod_%s/' %(env,str(norm),k,mod)
            os.makedirs(path, exist_ok=True)  #create directory when non-existing
            pickle.dump([auc_learn, auc_reward, learn_curve, reward_curve], \
                open("results/grid/%s/norm_%s/k_%s/mod_%s/params_%s.pkle" \
                %(env,str(norm),k,mod,params),"wb"))

    elapsed = time.time() - t
    print('Time elapsed: %f' % (elapsed))
    sys.stdout.flush()


if __name__ == '__main__':
    main(int(sys.argv[1]),sys.argv[2],int(sys.argv[3]),\
        int(sys.argv[4]),bool(int(sys.argv[5])),int(sys.argv[6]))
