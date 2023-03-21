##################################################################
# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
#
# Description: searches over outlined parameter space for model
# Splits search into parameter sweep chunks
# Saves states, AUC and curves for learning and reward in varying 
# levels of domain complexity
#
# Name: opal/complexity_rl.py
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
sys.path.insert(1, os.path.join(sys.path[0], '../standard_rl/'))
import learn


def main(job,env_base,n_trials,n_states,split):
    """ 
    Run specified parameter combinations for desired environment.
    Saves data to specified environment

    Inputs:
    job         - idx of parameter combination to try
    env_base    - "80"
    n_trials    - number of trials for learning
    n_states    - number of states to be averaging accuracy over
    """
    t = time.time()  # each level is about 30 min
    
    par_key = "RL"
    these_params = params.get_params(job,split,par_key)

    root_me = "revisions/RL/"
    n_levels = 3 #6 # number of levels of complexity

    r_mag = 1
    l_mag = 0
    v0 = 0.5*(r_mag + l_mag)

    # compute AUC for each level
    for level in np.arange(n_levels):

        level = level+3

        # get environment name
        # if level == 0:
        #     n_opt = ""                    # no number for base environment
        # else:   
        #     n_opt = "_" + str(level + 2)  # offset by 2
        # env = env_base + n_opt

        # just run highest and lowest complexity

        # use the more flexible environment calls
        diff = 10
        env = "%s_%d_%d" %(env_base,diff,level)
        print(env)

        # if random seeds exist, load
        # save random seeds
        seed_path = 'results/%s/trials%d_sims%d/' %(root_me,n_trials,n_states)
        seed_ext = '%s_rnd_seeds.pkle' %(env)
        path = './' + seed_path
        if os.path.exists(path + seed_ext):
            rnd_seeds = pickle.load(open(seed_path + seed_ext,"rb"))
            print("reused saved seeds")
        # if not, generate new random seed for each simulation 
        # to be shared across differences in simulation
        else:
            rnd_seeds = np.random.randint(1,100000,n_states)
            os.makedirs(path, exist_ok=True)  #create directory when non-existing
            pickle.dump(rnd_seeds, open(seed_path + seed_ext,"wb"))

        # do the thing
        for group in these_params:
            # get args
            pars = tuple(group[0:2]) #alpha,beta 

            # learn the things
            # params,n_states,n_trials,v0=0.0,env = "rich",r_mag = 1, l_mag = 0, rnd_seeds = None
            states = learn.simulate(pars,n_states,n_trials,v0=v0,env=env,r_mag=r_mag,l_mag=l_mag,rnd_seeds=rnd_seeds)

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
            path_ext = 'results/%s/trials%d_sims%d/%s/' \
                    %(root_me,n_trials,n_states,env)
            path = './' + path_ext
            print(path)
            os.makedirs(path, exist_ok=True)  #create directory when non-existing

            # save according to params
            pickle.dump([auc_learn, auc_reward, learn_curve, reward_curve, rnd_seeds], \
                    open(path_ext + "params_%s.pkle" %(str(pars)),"wb"))

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
    main(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]))
