##################################################################
# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
#
# Description: searches over outlined parameter space for learning.py
# Splits search into parameter sweep chunks
# Saves states, AUC and curves for learning and reward
#
# Name: bogacz/grid_search.py
##################################################################

import pickle
import os
import sys
import scipy.stats as stats
import numpy as np
from sklearn import metrics
import time

# Dependences
from bogacz import Bogacz
import learningbogacz as learning

sys.path.insert(1, os.path.join(sys.path[0], '../helpers'))
import params


def main(job,env_base,n_trials,n_states,split):
    """ 
    Run specified parameter combinations for desired environment.
    Saves data to specified environment

    Inputs:
    job         - idx of parameter combination to try
    env         - specified by environments.py
    n_trials    - number of trials for learning
    n_states    - number of states to be averaging accuracy over
    split       - number of splits for parameter combinations
    thresh      - threshold to start modulation
    """

    t = time.time()
    par_key = "ALU2"
    these_params = params.get_params(job,split,par_key)
    # ks = params.get_ks(par_key)
    ks = np.array([0.,1.])
    full = False     # give model full info?
    n_levels = 5    # TODO: CAHNGE BACK TO 8

    # compute AUC for each level
    for level in np.arange(n_levels):

        # get environment name
        diff = 10
        env = "%s_%d_%d" %(env_base,diff,level + 2)
        print(env)

        # load random seeds from opal sims
        # if random seeds exist, load
        # save random seeds
        seed_path = '../opal/sims/results/simplemod/Bayes-SA/rmag_1_lmag_0_mag_1/phi_1.0/usestd_False/exp_val_False/anneal_True_T_10/trials500_sims5000/'
        seed_ext = '%s_rnd_seeds.pkle' %(env)
        if os.path.exists(seed_path + seed_ext):
            rnd_seeds = pickle.load(open(seed_path + seed_ext,"rb"))
            print("reused saved seeds")
        # if not, generate new random seed for each simulation 
        # to be shared across differences in simulation
        else:
            err = 'could not find seed: ' + path
            raise Exception(err)
        

        for group in these_params:

            for k in ks:

                # get args
                pars = tuple(group[0:4])

                # get appropriate mod arg
                if k == 0:
                    mod = "constant"
                else:
                    mod = "value"

                # learn the things
                states = learning.simulate(pars,n_states,n_trials,\
                    env=env,policy="softmax", D=0.5, mod=mod,k=k,\
                    rnd_seeds=rnd_seeds,full=full)

                
                ### Post-processing ###
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
                if full:
                    add_on = "_FULL"
                else:
                    add_on = ""
                path = './results/complexity_%d_%d%s/%s/%s/k_%s/mod_%s/' %(n_trials,n_states,add_on,env_base,env,k,mod)
                os.makedirs(path, exist_ok=True)  #create directory when non-existing
                pickle.dump([auc_learn, auc_reward, learn_curve, reward_curve], \
                    open("results/complexity_%d_%d%s/%s/%s/k_%s/mod_%s/params_%s.pkle" \
                    %(n_trials,n_states,add_on,env_base,env,k,mod,str(np.round(pars,5))),"wb"))

                elapsed = time.time() - t
                print('k: %d. Time elapsed: %f' % (k,elapsed))
                sys.stdout.flush()

            elapsed = time.time() - t
            print('Param complete. Time elapsed: %f' % (elapsed))
            sys.stdout.flush()

    elapsed = time.time() - t
    print('Total Time elapsed: %f' % (elapsed))
    sys.stdout.flush()


if __name__ == '__main__':
    main(int(sys.argv[1]),sys.argv[2],int(sys.argv[3]),\
        int(sys.argv[4]),int(sys.argv[5]))
