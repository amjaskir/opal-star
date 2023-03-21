##################################################################
# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
#
# Decription: Compare effects of DA modulation in bayesian opal model
#
# Name: opal/hist_grid_bayes.py
##################################################################

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import scipy.stats as stats
from sklearn import metrics

# get my helper modules
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../../helpers'))
import params
import aucplotter

sys.path.insert(1, os.path.join(sys.path[0], '../../model_cmp'))
import filter

def getAUC(learn_curve,reward_curve,n_trials):
    [avg_sm,sem_sm] = learn_curve
    [avg_r,sem_r] = reward_curve

    # calc AUC of both
    x = np.arange(n_trials)
    auc_learn = metrics.auc(x,avg_sm[0:n_trials])
    auc_reward = metrics.auc(x,avg_r[0:n_trials])

    return auc_learn, auc_reward

def get_filtered_params(pltn):
    # get parameters that are in the top X% for all environments 
    # and complexities for the given time horizon
    param_path = "../../model_cmp/filter_0.10/revisions/trials0_%d/union.pkle" %(pltn)
    res = pickle.load(open(param_path,"rb"))
    param_combo = res["SACrit_BayesAnneal"]["80"]["0"]  # 'SACrit_BayesAnneal', 'SACrit_BayesAnneal_mod', complexity and level agnostic
    return param_combo

def get_ranked_params(pltn):
    # get parameters that are in the top X% for all environments 
    # and complexities for the given time horizon
    param_path = "../../model_cmp/filter_0.10/revisions/trials0_%d/auc_order.pkle" %(pltn)
    res = pickle.load(open(param_path,"rb")) #index is res[level][model]
    return res


def main(env_base,n_trials,n_states,pltn,crit,use_var,decay,gamma,pgrad,variant=None):
    """
    Graphs data outputed by grid_search.py into histograms

    env_base  	environment specified by environments.py
    n_trials, n_states	specifies simulation
    pltn		number of trials to plot
    variant		saving folder name
    """
    #######################################################
    # this is copied from complexity_revisions.py
    # Various settings - restriction, full, anneal
    # specify save destination

    if crit == "Bayes-SA":
        par_key = "bayesCrit"
        param_comb = params.get_params_all(par_key)
        alpha_as = np.unique(np.array(param_comb)[:,1])[::2] # list of all unique alpha_as
        betas = np.unique(np.array(param_comb)[:,2])[::2] 	# list of all unique betas
        ks = np.array([0.])
        # ks = params.get_ks(par_key)[1:] 
    else:
        par_key = "rlCrit"
        param_comb = params.get_params_all(par_key)
        alpha_cs = np.unique(np.array(param_comb)[:,1])[::2] # list of all unique alpha_cs
        alpha_as = np.unique(np.array(param_comb)[:,1])[::2] # list of all unique alpha_as
        betas = np.unique(np.array(param_comb)[:,2])[::2] 	# list of all unique betas
        ks = np.array([20])
        # ks = params.get_ks(par_key)[1:] 

    # only used filtered params 
    hist_version = "ranked" # "ranked", "filter"
    if hist_version == "filter":
        param_comb = get_filtered_params(pltn)
        filt_preface = "filt_"
    elif hist_version == "ranked":
        ranked_comb = get_ranked_params(pltn)
        param_comb = None
        filt_preface = "rank_"
    else:
        filt_preface = ""

    print("PARAMS")
    print(param_comb)

    # for ranked paths
    ranked_paths = { "RL": "results/revisions/RL/trials1000_sims1000/",
			 "UCB": "results/revisions/UCB/trials1000_sims1000/",
             "SACrit_BayesAnneal": "results/revisions/SA/rmag_1_lmag_0_mag_1/phi_1.0/anneal_True_T_10/use_var_True/decay_to_prior_False_gamma_10/pgrad_False/trials1000_sims1000/",
             "NoHebb": "results/revisions/SA/rmag_1_lmag_0_mag_1/phi_1.0/anneal_True_T_10/use_var_True/decay_to_prior_False_gamma_10/pgrad_False/trials1000_sims1000/",
			 "Bmod": "results/revisions/SA/rmag_1_lmag_0_mag_1/phi_1.0/anneal_True_T_10/use_var_True/decay_to_prior_False_gamma_10/pgrad_False/trials1000_sims1000/"}


    root_me = "revisions/"
    n_levels = 5 # number of levels of complexity

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
    T = 10 #10, 50, 100
    base = "anneal_%r_T_%d/use_var_%r/" %(anneal,T,use_var)
    root_me = root_me + base

    # decay to prior?
    base = "decay_to_prior_%r_gamma_%d/" %(decay,gamma)
    root_me = root_me + base

    # policy gradient
    base = "pgrad_%r/" %(pgrad)
    root_me = root_me + base

    # hebb = True	# non hebbian term
    # base = "hebb_%r/" %(hebb)
    # root_me = root_me + base

    #######################################################

    mod = "mod_beta"

    # # complexity graphs x ks
    # plt.rcParams.update({'font.size': 8})
    # fig_main, ax_main = plt.subplots(1,len(ks))
    # fig_main.set_size_inches(22, 17)
    # fig_main.tight_layout(pad=4.0, w_pad=4.0, h_pad=4.0)


    #create directory for complex graphs when non-existing
    save_path = "results/%s/trials%d_sims%d/%s/ntrials%d/%s/"\
            %(root_me,n_trials,n_states,env_base,pltn,mod)
    if variant is not None:
        save_path = save_path + variant + "/"
    os.makedirs(save_path, exist_ok=True)
    save_path = save_path + filt_preface

    # overall avg AUC for each k
    avg_auc_mod = np.zeros((len(ks),n_levels))
    sem_auc_mod = np.zeros((len(ks),n_levels))

    for k_idx,k in enumerate(ks):
        # track the mean diff in AUC between balanced/modulated
        # matched by parameters across levels
        mean_learn 	= np.zeros(n_levels)
        std_learn	= np.zeros(n_levels)
        mean_reward = np.zeros(n_levels)
        std_reward	= np.zeros(n_levels)

        for level in np.arange(n_levels):
            # just run highest and lowest complexity
            level_idx = level
            env = "%s_%d_%d" %(env_base,10,level_idx+2)
            print("Full Env: " + env)
            sys.stdout.flush()

            # NOTE - path_base doesn't need to account for variant since random seeds fixed
            # This means exact same baseline across variants with same critic
            path_base = "./results/%strials%d_sims%d/%s/k_0/mod_constant/" \
                %(root_me,n_trials,n_states,env) 
            path = "./results/%strials%d_sims%d/%s/k_%s/%s/" \
                %(root_me,n_trials,n_states,env,k,mod)

            # if looking at variant, compare to the k modulated version
            if variant is not None:
                path_base = path

            # TODO: fix this saving issue, added mod_beta to path and now saving incorrectly
            # save in separate folder for the specified pltn
            path_minus_mod = "./results/%strials%d_sims%d/%s/k_%s/" \
                %(root_me,n_trials,n_states,env,k)
            save_pltn = path_minus_mod + "ntrials" + str(pltn) + "/" + mod + "/"
            if variant is not None:
                save_pltn = save_pltn + variant + "/"
            os.makedirs(save_pltn, exist_ok=True)
            save_pltn = save_pltn + filt_preface

            auc_diff_learn = []
            auc_diff_reward = []
            auc_abs_diff_learn = []
            auc_abs_diff_reward = []

            auc_learn_modT = []
            auc_reward_modT = []
            auc_learn_modF = []
            auc_reward_modF = []

            # save parameters to color diff by AUC graphs
            color_beta = []
            color_alphac = []
            color_alphaa = []
            color_orig = []
            titles = ["Beta","Alpha A","Alpha C"]
            saveas = ["diff_by_auc_beta","diff_by_auc_alphaa","diff_by_auc_alphac"]
            maps = ["plasma","plasma","plasma"]	# use coolwarm for boolean

            # plt.rcParams.update({'font.size': 12})
            # fig_curv, ax_curv = plt.subplots(len(alpha_as),len(betas))
            # fig_curv.set_size_inches(22*1.5, 17*1.5)
            # fig_curv.tight_layout(pad=2.0, w_pad=2.0, h_pad=2.0)

            if hist_version == "ranked":
                print("IN RANKED")
                param_comb = ranked_comb[str(level+2)]["SACrit_BayesAnneal_mod"] # always comparing against DAmod
                base_comb = ranked_comb[str(level+2)][variant] # SACrit_BayesAnneal, UCB, or RL

            for pidx, par in enumerate(param_comb):
                
                # check pidx relative to baseline model length
                if hist_version == "ranked":
                    try:
                        base_pars = base_comb[pidx]
                    except:
                        print("completed main loop")
                        break # no more base parameters


                # get params
                if crit == "Bayes-SA":
                    str_params = str(par[1:3])
                else: # SA
                    str_params = str(par[0:3])

                # handle parameter ish
                alpha_c, alpha_a, beta = par[0:3]

                if variant is None:
                    base_addition  = "params_"
                else:
                    base_addition = variant + "/params_"


                # if os.path.exists(path_base + "params_" + str_params + ".pkle"): #unmodulated
                if os.path.exists(path + "params_" + str_params + ".pkle"):

                    # save color information
                    color_alphac.append(alpha_c)
                    color_alphaa.append(alpha_a)
                    color_beta.append(beta)	

                    # if opt params alpha = .8, beta = 1, mark
                    # color orig files
                    if (float(alpha_a) == .8) and (float(beta) == 1.):
                        color_orig.append(0) # change to 1 and replace above
                    else:
                        color_orig.append(0)

                    # get modulated data
                    str_path = path + "params_" + str_params + ".pkle"
                    _, _, learn_curveT, reward_curveT, _ = \
                        pickle.load(open(str_path,"rb")) # use modulated
                    auc_learnT, auc_rewardT = getAUC(learn_curveT,reward_curveT,pltn)

                    # get unmodulated data 
                    # change str_params and path_base here if performing ranking analysis
                    if hist_version == "ranked":
                        str_path_base = filter.get_full_path(variant,env_base,level+2,k,base_pars,ranked_paths)
                    else:
                        str_path_base = path_base + base_addition + str_params + ".pkle"
                    _, _, learn_curveF, reward_curveF, _ = \
                    pickle.load(open(str_path_base,"rb"))
                    auc_learnF, auc_rewardF = getAUC(learn_curveF,reward_curveF,pltn)
                    print("param: " + str_params)
                    print("path modulated: " + str_path)
                    print("path base: " + str_path_base)
                    print("AUC: " + str(auc_learnF))

                    # save perf difference, same parameters, modulated (T) or balanced (F)
                    auc_diff_learn.append((auc_learnT - auc_learnF)/auc_learnF)
                    auc_diff_reward.append((auc_rewardT - auc_rewardF)/auc_rewardF)

                    auc_abs_diff_learn.append((auc_learnT - auc_learnF))
                    auc_abs_diff_reward.append((auc_rewardT - auc_rewardF))

                    auc_learn_modT.append(auc_learnT)
                    auc_learn_modF.append(auc_learnF)
                    auc_reward_modT.append(auc_rewardT)
                    auc_reward_modF.append(auc_rewardF)

                    # save learning curves for parameter
                    aucplotter.learn_curve(learn_curveT,auc_learnT,learn_curveF,auc_learnF,str_params,save_pltn,\
                        w_auc=True,pltn=pltn)

                    # # save learning curves in larger grid figure
                    # which_a = np.where(alpha_as == alpha_a)[0][0]
                    # which_b = np.where(betas == beta)[0][0]
                    # this_ax = ax_curv[which_a,which_b]
                    # plt.rcParams.update({'font.size': 8})
                    # aucplotter.learn_curve(learn_curveT,auc_learnT,learn_curveF,auc_learnF,str_params,save_pltn,\
                    #     w_auc=False,pltn=pltn,ax=this_ax)

                else:
                    tried = str(path + "params_" + str_params + ".pkle")
                    print(tried)
                    print("missing params", str_params)
                    sys.stdout.flush()

            # # # save larger curve grid
            # plt.rcParams.update({'font.size': 8})
            # plt.savefig(save_pltn + "allcurves.png",dpi = 500)
            # plt.close()

            colors = (color_beta,color_alphaa,color_alphac)

            # get mean and std of auc diff for the level
            mean_learn[level] 	= np.mean(auc_diff_learn)
            std_learn[level]	= stats.sem(auc_diff_learn)
            mean_reward[level]	= np.mean(auc_diff_reward)
            std_reward[level]	= stats.sem(auc_diff_reward)

            # save mean and std of mod auc perf
            avg_auc_mod[k_idx,level] = np.mean(auc_learn_modT)
            sem_auc_mod[k_idx,level] = stats.sem(auc_learn_modT)

            # get max diff and max no mod auc and their respective params
            # save to file
            selectedParams =  save_pltn + "exampleParams.txt"
            maxdiff = (np.max(auc_diff_learn) == auc_diff_learn)
            maxdiffparams = np.where(maxdiff == 1)[0][0]
            maxauc = (np.max(auc_learn_modF) == auc_learn_modF)
            maxaucparams = np.where(maxauc == 1)[0][0]
            maxdiff = maxdiff*1 # convert to int
            maxauc = maxauc*2
            example_curves = maxdiff + maxauc # convert to color map

            # save to text for reference
            with open(selectedParams, "w") as text_file:
                saveme = "MaxDiffParams: %s" % str(param_comb[maxdiffparams])
                text_file.write(saveme)
                saveme = "MaxAUCParams: %s" % str(param_comb[maxaucparams])
                text_file.write(saveme)


            # Graph AUC for each level
            if variant is None:
                what = "mod"
            else:
                what = variant
            aucplotter.graph_learn(auc_learn_modT,auc_learn_modF,save_pltn,what)
            aucplotter.graph_reward(auc_reward_modT,auc_reward_modF,save_pltn,what)

            # difference in AUC
            aucplotter.graph_diff(auc_diff_learn,save_pltn,"Learning",what)
            aucplotter.graph_diff(auc_diff_reward,save_pltn,"Reward",what)
            aucplotter.graph_diff_by_AUC(np.array(auc_learn_modF),np.array(auc_diff_learn),\
                colors,titles,saveas,maps,\
                save_pltn,"Learning",what)
            aucplotter.graph_diff_by_AUC(np.array(auc_reward_modF),np.array(auc_diff_reward),\
                colors,titles,saveas,maps,\
                save_pltn,"Reward",what)
            aucplotter.graph_diff_by_AUC(np.array(auc_learn_modF),np.array(auc_diff_learn),\
                colors,titles,saveas,maps,\
                save_pltn,"LearningDotted",what,color_orig=example_curves)
            aucplotter.graph_diff_by_AUC(np.array(auc_reward_modF),np.array(auc_diff_reward),\
                colors,titles,saveas,maps,\
                save_pltn,"RewardDotted",what,color_orig=example_curves)

            # now for absolute difference
            aucplotter.graph_diff(auc_abs_diff_learn,save_pltn,"Learning",what,abs_diff=True)
            aucplotter.graph_diff(auc_abs_diff_reward,save_pltn,"Reward",what,abs_diff=True)
            aucplotter.graph_diff_by_AUC(np.array(auc_learn_modF),np.array(auc_abs_diff_learn),\
                colors,titles,saveas,maps,\
                save_pltn,"Learning",what,abs_diff=True)
            aucplotter.graph_diff_by_AUC(np.array(auc_reward_modF),np.array(auc_abs_diff_reward),\
                colors,titles,saveas,maps,\
                save_pltn,"Reward",what,abs_diff=True)
            aucplotter.graph_diff_by_AUC(np.array(auc_learn_modF),np.array(auc_abs_diff_learn),\
                colors,titles,saveas,maps,\
                save_pltn,"LearningDotted",what,abs_diff=True,color_orig=example_curves)
            aucplotter.graph_diff_by_AUC(np.array(auc_reward_modF),np.array(auc_abs_diff_reward),\
                colors,titles,saveas,maps,\
                save_pltn,"RewardDotted",what,abs_diff=True,color_orig=example_curves)

            # TODO: put max curves     
            # I want the above curves to return their figs
            # basically, I can pass axes. if axes = none, do what I already do. if axes defined, use that. 
            # functions to modify: graph diff and graph diff by AUC. for by auc only iterate through colors if axes aren't specified
             

            # AUC according to alphas, high alpha should have decrease in AUC
            aucplotter.auc_by_alpha(np.array(auc_learn_modF),color_alphaa,save_pltn,"Learning")
            aucplotter.auc_by_alpha(np.array(auc_reward_modF),color_alphaa,save_pltn,"Reward")

        if variant == "lrate":
            ext = "k%s" %(k)
            ext = ext.replace(".", "")
        else:
            ext = "k%s" %(int(k))


        # plot and save
        # for learning
        xaxis = np.arange(n_levels)
        plt.rcParams.update({'font.size': 22})
        fig, ax = plt.subplots()
        ax.errorbar(xaxis,mean_learn,yerr=std_learn)
        plt.ylabel("AUC (Mod - Bal)/Bal")
        plt.xlabel("Complexity")
        plt.title("Learning")
        plt.tight_layout()
        plt.savefig(save_path + ext + "learning")
        plt.close()

        # for reward
        xaxis = np.arange(n_levels)
        plt.rcParams.update({'font.size': 22})
        fig, ax = plt.subplots()
        ax.errorbar(xaxis,mean_learn,yerr=std_learn)
        plt.ylabel("AUC (Mod - Bal)/Bal")
        plt.xlabel("Complexity")
        plt.title("Reward")
        plt.tight_layout()
        plt.savefig(save_path + ext + "reward")
        plt.close()


        # ## add to the giant one
        # which_k = np.where(ks == k)[0][0]
        # xaxis = np.arange(n_levels)
        # ax_main[which_k].errorbar(xaxis,mean_learn,yerr=std_learn)
        # ax_main[which_k].set_title("k=%d" %k)

        print("k %d done" %(k))
        sys.stdout.flush()

    # plt.rcParams.update({'font.size': 22})
    # plt.savefig(save_path + "learning", dpi = 400)
    # plt.close()


    # plot avg auc by k and complexity level
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 22})
    xaxis = np.arange(n_levels)
    for k_idx,k in enumerate(ks):
        offset = k_idx/10
        plt.errorbar(xaxis+offset,avg_auc_mod[k_idx,:],yerr=sem_auc_mod[k_idx,:],label=str(k),linewidth=1.0)
    plt.ylabel("Avg AUC Mod")
    plt.xlabel("Complexity")
    plt.title("Learning")
    plt.savefig(save_path + "auc_by_k", dpi = 400)
    plt.close()




if __name__ == '__main__':
    if len(sys.argv) < 11:
        main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),\
            sys.argv[5],bool(int(sys.argv[6])),bool(int(sys.argv[7])),
            int(sys.argv[8]),bool(int(sys.argv[9])))
    else:
        # use a variant of the opal code
        main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),\
            sys.argv[5],bool(int(sys.argv[6])),bool(int(sys.argv[7])),
            int(sys.argv[8]),bool(int(sys.argv[9])),sys.argv[10])





