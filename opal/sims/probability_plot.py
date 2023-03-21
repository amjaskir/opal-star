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

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# get my helper modules
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../../helpers'))
import params


def getAUC(learn_curve,reward_curve,n_trials):
    [avg_sm,sem_sm] = learn_curve
    [avg_r,sem_r] = reward_curve

    # calc AUC of both
    x = np.arange(n_trials)
    auc_learn = metrics.auc(x,avg_sm[0:n_trials])
    auc_reward = metrics.auc(x,avg_r[0:n_trials])

    return auc_learn, auc_reward

def main(pltn):
    """ 
    Run specified parameter combinations for desired environment.
    Saves data to specified environment

    Inputs:
    job         - idx of parameter combination to try
    n_trials    - number of trials for learning
    n_states    - number of states to be averaging accuracy over
    split       - number of splits for parameter combinations
    """
    t = time.time()
    root_me = "complexity"
    n_levels = 8
    levels = np.arange(n_levels) + 2 # number of levels of complexity
    prob_base = np.arange(.2,1.0,.1) # ranges of opt_prob
    n_probs = len(prob_base)
    step_sz = 10    #step size in %

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
    n_trials = 500
    n_states = 5000

    # constant parameters for learning
    v0 = 0.0
    crit = "Bayes"
    norm = True
    ks = np.array([20])
    # ks = np.array([1,2,10,20,100,200])
    # ks = np.array([20])

    # DA modulation variations
    # mod = "beta"
    mod = "beta_50_phi3"
    # mod = "beta_var"

    par_key = "extrange"
    param_comb = params.get_params_all(par_key)
    #######################################################

    for k in ks:

        save_me = 'results/%s_%d_%d/summaries/probability_space/mod_%s/ntrials%d/' %(root_me,n_trials,n_states,mod,pltn)
        os.makedirs(save_me, exist_ok=True)  #create directory when non-existing
        save_me = save_me + "k%d_" %(int(k))
        reprocess = False 

        # if already exists
        if os.path.exists(save_me + "res.pkle") and not reprocess:
            res = pickle.load(open(save_me + "res.pkle","rb"))

            mean_learn = res["mean_learn"] 
            sem_learn = res["sem_learn"] 
            mean_reward = res["mean_reward"]  
            sem_reward =res["sem_reward"] 

            mean_learn_abs = res["mean_learn_abs"]
            sem_learn_abs = res["sem_learn_abs"]
            mean_reward_abs = res["mean_reward_abs"]
            sem_reward_abs = res["sem_reward_abs"]

        else:

            # first check if the data exists
            # normalized difference
            mean_learn  = np.zeros([n_levels,n_probs])
            sem_learn   = np.zeros([n_levels,n_probs])
            mean_reward = np.zeros([n_levels,n_probs])
            sem_reward  = np.zeros([n_levels,n_probs])

            # absolute difference
            mean_learn_abs  = np.zeros([n_levels,n_probs])
            sem_learn_abs   = np.zeros([n_levels,n_probs])
            mean_reward_abs = np.zeros([n_levels,n_probs])
            sem_reward_abs  = np.zeros([n_levels,n_probs])

            for l_idx, level in enumerate(levels):
                print("level: %s" %str(level))
                for p_idx, opt_prob in enumerate(prob_base):
                    print("opt_prob: %s" %str(opt_prob))

                    # get environment name
                    env = "%d_%d_%d" %(opt_prob*100,step_sz,level)

                    path_base = "results/%s_%d_%d/%s/k_0/mod_constant/" \
                        %(root_me,n_trials,n_states,env)
                    path = 'results/%s_%d_%d/%s/k_%s/mod_%s/' \
                                %(root_me,n_trials,n_states,env,k,mod)

                    auc_diff_learn = []
                    auc_diff_reward = []
                    auc_abs_diff_learn = []
                    auc_abs_diff_reward = []

                    # get AUC differences for each parameter
                    for par in param_comb:

                        par2 = par[1:3]
                        print("par: %s" %str(par2))

                        # handle parameter ish
                        alpha_a, beta = par2

                        # load data if it exists
                        if os.path.exists(path + "params_" + str(par2) + ".pkle"):
                            # get modulated data
                            _, _, learn_curve_mod, reward_curve_mod, rnd_seed = \
                            pickle.load(open(path + "params_" + str(par2) + ".pkle","rb"))
                            auc_learn_mod, auc_reward_mod = \
                            getAUC(learn_curve_mod,reward_curve_mod,pltn)

                            _, _, learn_curve_bal, reward_curve_bal, rnd_seed = \
                            pickle.load(open(path_base + "params_" + str(par2) + ".pkle","rb"))
                            auc_learn_bal, auc_reward_bal = \
                            getAUC(learn_curve_bal,reward_curve_bal,pltn)

                            auc_diff_learn.append((auc_learn_mod - auc_learn_bal)/auc_learn_bal)
                            auc_diff_reward.append((auc_reward_mod - auc_reward_bal)/auc_reward_bal)

                            auc_abs_diff_learn.append((auc_learn_mod - auc_learn_bal))
                            auc_abs_diff_reward.append((auc_reward_mod - auc_reward_bal))
                        else:
                            tried = str(path + "params_" + str(par2) + ".pkle")
                            print("missing params: %s" + tried)
                            # raise Exception("env:%s, missing params: %s" %(env,str(par2)))

                    # get average AUC differences for all parameters
                    mean_learn[l_idx,p_idx]   = np.mean(auc_diff_learn)
                    sem_learn[l_idx,p_idx]    = stats.sem(auc_diff_learn)
                    mean_reward[l_idx,p_idx]  = np.mean(auc_diff_reward)
                    sem_reward[l_idx,p_idx]   = stats.sem(auc_diff_reward)

                    mean_learn_abs[l_idx,p_idx]   = np.mean(auc_abs_diff_learn)
                    sem_learn_abs[l_idx,p_idx]    = stats.sem(auc_abs_diff_learn)
                    mean_reward_abs[l_idx,p_idx]  = np.mean(auc_abs_diff_reward)
                    sem_reward_abs[l_idx,p_idx]   = stats.sem(auc_abs_diff_reward)

             # save all that information for later
            res = {}
            res["mean_learn"] = mean_learn
            res["sem_learn"] = sem_learn
            res["mean_reward"] = mean_reward
            res["sem_reward"] = sem_reward

            res["mean_learn_abs"] = mean_learn_abs
            res["sem_learn_abs"] = sem_learn_abs
            res["mean_reward_abs"] = mean_reward_abs
            res["sem_reward_abs"] = sem_reward_abs

            pickle.dump(res,open(save_me + "res.pkle","wb"))

        # plot for each k
        plt.rcParams.update({'font.size': 44})
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(22, 17)
        for l_idx,level in enumerate(levels):
            ax1.errorbar(prob_base,mean_learn[l_idx,:],yerr=sem_learn[l_idx,:],label=level)
        plt.ylabel("AUC (Mod - Bal)/Bal")
        plt.xlabel("Best option")
        plt.title("Learning")
        ax1.legend()
        plt.tight_layout()
        plt.savefig(save_me + "learning")
        plt.close()

        plt.rcParams.update({'font.size': 44})
        fig2, ax2 = plt.subplots()
        fig2.set_size_inches(22, 17)
        for l_idx,level in enumerate(levels):
            ax2.errorbar(prob_base,mean_reward[l_idx,:],yerr=sem_reward[l_idx,:],label=level)
        plt.ylabel("AUC (Mod - Bal)/Bal")
        plt.xlabel("Best option")
        plt.title("Reward")
        ax2.legend()
        plt.tight_layout()
        plt.savefig(save_me + "reward")
        plt.close()

        plt.rcParams.update({'font.size': 44})
        fig3, ax3 = plt.subplots()
        fig3.set_size_inches(22, 17)
        for l_idx,level in enumerate(levels):
            ax3.errorbar(prob_base,mean_learn_abs[l_idx,:],yerr=sem_learn_abs[l_idx,:],label=level)
        plt.ylabel("AUC Mod - Bal")
        plt.xlabel("Best option")
        plt.title("Learning")
        ax3.legend()
        plt.tight_layout()
        plt.savefig(save_me + "learning_abs")
        plt.close()

        plt.rcParams.update({'font.size': 44})
        fig4, ax4 = plt.subplots()
        fig4.set_size_inches(22, 17)
        for l_idx,level in enumerate(levels):
            ax4.errorbar(prob_base,mean_reward_abs[l_idx,:],yerr=sem_reward_abs[l_idx,:],label=level)
        plt.ylabel("AUC Mod - Bal")
        plt.xlabel("Best option")
        plt.title("Reward")
        ax4.legend()
        plt.tight_layout()
        plt.savefig(save_me + "reward_abs")
        plt.close()


if __name__ == '__main__':
    main(int(sys.argv[1]))
