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
# Name: opal/complexity.py
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


def main(job,env_base,n_trials,n_states,split,crit,use_var,decay,gamma,pgrad,variant=None):
    """ 
    Run specified parameter combinations for desired environment.
    Saves data to specified environment

    Inputs:
    job         - idx of parameter combination to try
    env_base    - "80"
    n_trials    - number of trials for learning
    n_states    - number of states to be averaging accuracy over
    split       - number of splits for parameter combinations
    variant     - variant of opal to be run
                  "flip", "bmod","lrate", "nohebb"
    """
    t = time.time()  # each level is about 30 min
    if crit == "Bayes-SA":
        par_key = "bayesCrit"
    else:
        par_key = "rlCrit"
    these_params = params.get_params(job,split,par_key)
    ks = np.array([0,20])

    #######################################################
    # Various settings - restriction, full, anneal
    # specify save destination

    # give model full info?
    # full = False
    # c_only = False
    # if full:
    #     if c_only == True:
    #         root_me = "full_c_only/" + root_me
    #     else:
    #         root_me = "full/" + root_me

    root_me = "revisions/"
    n_levels = 3 #6 # number of levels of complexity

    # constant parameters for learning
    crit = crit
    norm = True
    root_me = root_me + crit + "/"

    # reward mag and loss mag
    # same for each option
    r_mag = 1
    l_mag = 0
    if crit == "Bayes-SA":
        v0 = np.array([0.,0.])
    else:
        v0 = 0.5*(r_mag + l_mag)
    mag = r_mag - l_mag
    base = "rmag_%d_lmag_%d_mag_%d/" %(r_mag,l_mag,mag)
    root_me = root_me + base

    # only modify sufficiently above 50% by phi*std
    # now coded that mod is only when sufficient
    phi = 1.0
    base = "phi_%.1f/" %(phi)
    root_me = root_me + base

    # anneal learning rate?
    anneal = True
    T = 10.0 #10, 100
    base = "anneal_%r_T_%d/use_var_%r/" %(anneal,T,use_var)
    root_me = root_me + base

    # decay to prior?
    base = "decay_to_prior_%r_gamma_%d/" %(decay,gamma)
    root_me = root_me + base

    # policy gradient
    base = "pgrad_%r/" %(pgrad)
    root_me = root_me + base

    hebb = True
    # var_arg specifies learning rate
    if variant == "nohebb":
        print("nohebb")
        hebb = False
        var_arg = None
    else:
        var_arg = variant

    # hebb = True	# non hebbian term
    # base = "hebb_%r/" %(hebb)
    # root_me = root_me + base

    #######################################################

    # compute AUC for each level
    for level in np.arange(n_levels):

        # get environment name
        # if level == 0:
        #     n_opt = ""                    # no number for base environment
        # else:   
        #     n_opt = "_" + str(level + 2)  # offset by 2
        # env = env_base + n_opt

        # use the more flexible environment calls
        diff = 10
        env = "%s_%d_%d" %(env_base,diff,level + 3)
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
            for k in ks:
                # if (level == 0) and (k == 0):
                #     continue
                # get args
                pars = tuple(group[0:3])
                print(group[0:3])

                # get appropriate mod arg
                if k == 0:
                    mod = "constant"
                else:
                    mod = "beta"

                # learn the things
                states = learning.simulate(pars,n_states,n_trials,v0=v0,crit=crit,hebb=hebb,\
                    env=env,mod=mod,k=k,norm=norm,mag=mag,rnd_seeds=rnd_seeds,\
                    anneal=anneal,T=T,use_var=use_var, pgrad=pgrad,\
                    decay_to_prior=decay, decay_to_prior_gamma=gamma,\
                    variant=var_arg,phi=phi,r_mag=r_mag,l_mag=l_mag)

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
                path_ext = 'results/%s/trials%d_sims%d/%s/k_%s/mod_%s/' \
                        %(root_me,n_trials,n_states,env,k,mod)
                if variant is not None:
                    path_ext = path_ext + variant + "/"
                path = './' + path_ext
                print(path)
                os.makedirs(path, exist_ok=True)  #create directory when non-existing

                # save according to appropriate params
                if crit == "Bayes-SA":
                    str_params = str(pars[1:])
                else: # SA
                    str_params = str(pars)
                pickle.dump([auc_learn, auc_reward, learn_curve, reward_curve, rnd_seeds], \
                        open(path_ext + "params_%s.pkle" %(str_params),"wb"))

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
    if len(sys.argv) < 12:
        main(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),\
            int(sys.argv[4]),int(sys.argv[5]),sys.argv[6],bool(int(sys.argv[7])),\
                bool(int(sys.argv[8])),int(sys.argv[9]),bool(int(sys.argv[10])))
    else:
        # use a variant of the opal code
        main(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),\
            int(sys.argv[4]),int(sys.argv[5]),sys.argv[6],bool(int(sys.argv[7])),\
                bool(int(sys.argv[8])),int(sys.argv[9]),bool(int(sys.argv[10])),\
                    sys.argv[11])
