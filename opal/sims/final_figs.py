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

def getAUC(learn_curve,reward_curve,n_trials):
    [avg_sm,sem_sm] = learn_curve
    [avg_r,sem_r] = reward_curve

    # calc AUC of both
    x = np.arange(n_trials)
    auc_learn = metrics.auc(x,avg_sm[0:n_trials])
    auc_reward = metrics.auc(x,avg_r[0:n_trials])

    return auc_learn, auc_reward


def main(n_trials,n_states,pltn):
    """
    Graphs data outputed by grid_search.py into histograms

    env_base  	environment specified by environments.py
    n_trials, n_states	specifies simulation
    pltn		number of trials to plot
    """

    env_bases = ["80","30"]
    variants = ["balanced","nohebb","bmod"] #controls to compare OpAL* with
    # variants = ["bmod"] #controls to compare OpAL* with

    #######################################################
    # This is copied from complexity.py
    par_key = "simplemod"
    param_comb = params.get_params_all(par_key)
    alpha_as = np.unique(np.array(param_comb)[:,1]) # list of all unique alpha_as
    betas = np.unique(np.array(param_comb)[:,2])	# list of all unique betas

    k = 20.0

    root_me = "simplemod/"
    n_levels = 2
    mod = "mod_beta"

    # constant parameters for learning
    crit = "Bayes-SA"
    norm = True	# normalization
    root_me = root_me + crit + "/"

    # reward mag and loss mag
    # same for each option
    r_mag = 1
    l_mag = 0
    v0 = 0.5*r_mag + .5*l_mag
    mag = r_mag - l_mag
    base = "rmag_%d_lmag_%d_mag_%d/" %(r_mag,l_mag,mag)
    root_me = root_me + base

    # only modify sufficiently above 50% by phi*std
    # now coded that mod is only when sufficient
    phi = 1.0
    base = "phi_%.1f/" %(phi)
    root_me = root_me + base

    # use_std for K MODULATION 
    use_std = False # this should always be false, it was a variant that I never explored
    base = "usestd_%r/" %(use_std)
    root_me = root_me + base

    # Use expected value (vs. beta mean) for mod
    exp_val = False # this should always be false, it was a variant that I never explored
    base = "exp_val_%r/" %(exp_val)
    root_me = root_me + base

    # anneal learning rate?
    anneal = True
    T = 10.0
    base = "anneal_%r_T_%d/" %(anneal,T)
    root_me = root_me + base

    #######################################################

    for variant in variants:
        print("starting variant: %s" %variant)

        control_AUC = {}
        opalstar_AUC = {}

        for env_base in env_bases:
            print("starting env: %s" %env_base)

            #create directory for complex graphs when non-existing
            save_path = "results/FINALFIGS/%strials%d_sims%d/%s/ntrials%d/%s/"\
                    %(root_me,n_trials,n_states,env_base,pltn,mod)
            save_path = save_path + variant + "/"
            os.makedirs(save_path, exist_ok=True)

            # track the mean diff in AUC between balanced/modulated
            # matched by parameters across levels
            mean_learn 	= np.zeros(n_levels)
            std_learn	= np.zeros(n_levels)
            mean_reward = np.zeros(n_levels)
            std_reward	= np.zeros(n_levels)

            for level in np.arange(n_levels):
                if level == 1:
                    level = 4
                print("level: %s" %level)
                sys.stdout.flush()

                env = "%s_%d_%d" %(env_base,10,level+2)
                print("env: %s" %env)

                # path_base is the control model
                if variant == "balanced":
                    path_base = "./results/%strials%d_sims%d/%s/k_0.0/mod_constant/" \
                        %(root_me,n_trials,n_states,env) 
                else:
                    path_base = "./results/%strials%d_sims%d/%s/k_%s/%s/%s/" \
                        %(root_me,n_trials,n_states,env,k,mod,variant)

                path = "./results/%strials%d_sims%d/%s/k_%s/%s/" \
                    %(root_me,n_trials,n_states,env,k,mod)

                # save in separate folder for the specified pltn
                # save_pltn = path + "ntrials" + str(pltn) + "/" + mod + "/"
                save_pltn = save_path + env + "/"
                os.makedirs(save_pltn, exist_ok=True)

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
                color_alphaa = []
                color_orig = []
                titles = ["Beta","Alpha A"]
                saveas = ["diff_by_auc_beta","diff_by_auc_alphaa"]
                maps = ["plasma","plasma"]	# use coolwarm for boolean

                plt.rcParams.update({'font.size': 12})
                fig_curv, ax_curv = plt.subplots(len(alpha_as),len(betas))
                fig_curv.set_size_inches(22*1.5, 17*1.5)
                fig_curv.tight_layout(pad=2.0, w_pad=2.0, h_pad=2.0)

                idx = 0
                for par in param_comb:
                    par2 = par[1:3]
                    # print("par: %s" %str(par2))

                    # handle parameter ish
                    alpha_a, beta = par2
                    if alpha_a not in alpha_as:
                        continue
                    if beta not in betas:
                        continue


                    if os.path.exists(path + "params_" + str(par2) + ".pkle"):

                        # save color information
                        color_alphaa.append(alpha_a)
                        color_beta.append(beta)	

                        # if opt params alpha = .8, beta = 1, mark
                        # color orig files
                        if (float(alpha_a) == .8) and (float(beta) == 1.):
                            color_orig.append(0) # change to 1 and replace above
                        else:
                            color_orig.append(0)
                        
                        if (idx == 0):
                            p1 = path + "params_" + str(par2) + ".pkle"
                            p2 = path_base + "params_" + str(par2) + ".pkle"
                            print("opal path: %s" %p1)
                            print("var path: %s" %p2)
                            idx = 1

                        # get modulated data
                        _, _, learn_curveT, reward_curveT, rnd_seed = \
                        pickle.load(open(path + "params_" + str(par2) + ".pkle","rb"))
                        auc_learnT, auc_rewardT = getAUC(learn_curveT,reward_curveT,pltn)

                        # get unmodulated data 
                        _, _, learn_curveF, reward_curveF, rnd_seed = \
                        pickle.load(open(path_base + "params_" + str(par2) + ".pkle","rb"))
                        auc_learnF, auc_rewardF = getAUC(learn_curveF,reward_curveF,pltn)

                        # save perf difference, same parameters, modulated (T) or balanced (F)
                        auc_diff_learn.append((auc_learnT - auc_learnF)/auc_learnF)
                        auc_diff_reward.append((auc_rewardT - auc_rewardF)/auc_rewardF)

                        auc_abs_diff_learn.append((auc_learnT - auc_learnF))
                        auc_abs_diff_reward.append((auc_rewardT - auc_rewardF))

                        auc_learn_modT.append(auc_learnT)
                        auc_learn_modF.append(auc_learnF)
                        auc_reward_modT.append(auc_rewardT)
                        auc_reward_modF.append(auc_rewardF)

                        # # save learning curves for parameter
                        aucplotter.learn_curve(learn_curveT,auc_learnT,learn_curveF,auc_learnF,par2,save_pltn,\
                            w_auc=False,pltn=pltn,ttle=False) #ylim=[0.5,.8]

                        # # save learning curves in larger grid figure
                        # which_a = np.where(alpha_as == alpha_a)[0][0]
                        # which_b = np.where(betas == beta)[0][0]
                        # this_ax = ax_curv[which_a,which_b]
                        # plt.rcParams.update({'font.size': 8})
                        # aucplotter.learn_curve(learn_curveT,auc_learnT,learn_curveF,auc_learnF,par2,save_pltn,\
                        #     w_auc=False,pltn=pltn,ax=this_ax)

                    else:
                        tried = str(path + "params_" + str(par2) + ".pkle")
                        print(tried)
                        print("missing params", str(par2))
                        sys.stdout.flush()

                # # # save larger curve grid
                # plt.rcParams.update({'font.size': 8})
                # plt.savefig(save_pltn + "allcurves.png",dpi = 500)
                # plt.close()

                colors = (color_beta,color_alphaa)

                # get mean and std of auc diff for the level
                mean_learn[level] 	= np.mean(auc_diff_learn)
                std_learn[level]	= stats.sem(auc_diff_learn)
                mean_reward[level]	= np.mean(auc_diff_reward)
                std_reward[level]	= stats.sem(auc_diff_reward)

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
                what = variant
                aucplotter.graph_learn(auc_learn_modT,auc_learn_modF,save_pltn,what)
                aucplotter.graph_reward(auc_reward_modT,auc_reward_modF,save_pltn,what)

                # difference in AUC
                aucplotter.graph_diff(auc_diff_learn,save_pltn,"Learning",what)
                aucplotter.graph_diff(auc_diff_reward,save_pltn,"Reward",what)
                aucplotter.graph_diff_by_AUC(np.array(auc_learn_modF),np.array(auc_diff_learn),\
                    colors,titles,saveas,maps,\
                    save_pltn,"Learning",what,color_orig=example_curves)
                aucplotter.graph_diff_by_AUC(np.array(auc_reward_modF),np.array(auc_diff_reward),\
                    colors,titles,saveas,maps,\
                    save_pltn,"Reward",what,color_orig=example_curves)

                # now for absolute difference
                aucplotter.graph_diff(auc_abs_diff_learn,save_pltn,"Learning",what,abs_diff=True)
                aucplotter.graph_diff(auc_abs_diff_reward,save_pltn,"Reward",what,abs_diff=True)
                aucplotter.graph_diff_by_AUC(np.array(auc_learn_modF),np.array(auc_abs_diff_learn),\
                    colors,titles,saveas,maps,\
                    save_pltn,"LearningDotted",what,abs_diff=True,color_orig=example_curves)
                aucplotter.graph_diff_by_AUC(np.array(auc_reward_modF),np.array(auc_abs_diff_reward),\
                    colors,titles,saveas,maps,\
                    save_pltn,"RewardDotted",what,abs_diff=True,color_orig=example_curves)

                # now for absolute difference - no example curves highlighted
                aucplotter.graph_diff(auc_abs_diff_learn,save_pltn,"Learning",what,abs_diff=True)
                aucplotter.graph_diff(auc_abs_diff_reward,save_pltn,"Reward",what,abs_diff=True)
                aucplotter.graph_diff_by_AUC(np.array(auc_learn_modF),np.array(auc_abs_diff_learn),\
                    colors,titles,saveas,maps,\
                    save_pltn,"Learning",what,abs_diff=True)
                aucplotter.graph_diff_by_AUC(np.array(auc_reward_modF),np.array(auc_abs_diff_reward),\
                    colors,titles,saveas,maps,\
                    save_pltn,"Reward",what,abs_diff=True)

                # AUC according to alphas, high alpha should have decrease in AUC
                aucplotter.auc_by_alpha(np.array(auc_learn_modF),color_alphaa,save_pltn,"Learning")
                aucplotter.auc_by_alpha(np.array(auc_reward_modF),color_alphaa,save_pltn,"Reward")

                # save env/level parameter performance
                control_AUC[env] = auc_learn_modF
                opalstar_AUC[env] = auc_learn_modT

            ext = "k%s" %(int(k))

            # plot and save
            # for learning
            xaxis = np.arange(n_levels) + 2
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
            xaxis = np.arange(n_levels) + 2
            plt.rcParams.update({'font.size': 22})
            fig, ax = plt.subplots()
            ax.errorbar(xaxis,mean_learn,yerr=std_learn)
            plt.ylabel("AUC (Mod - Bal)/Bal")
            plt.xlabel("Complexity")
            plt.title("Reward")
            plt.tight_layout()
            plt.savefig(save_path + ext + "reward")
            plt.close()

            plt.rcParams.update({'font.size': 22})
            plt.savefig(save_path + "learning", dpi = 400)
            plt.close()

        # compare across env performance
        for level in np.arange(n_levels):
            print("cross env analysis")
            env_rich = "80_%d_%d" %(10,level+2)
            env_lean = "30_%d_%d" %(10,level+2)

            # avg auc across env per env
            meanControl = np.mean([control_AUC[env_rich],control_AUC[env_lean]],axis=0)
            meanOpalStar = np.mean([opalstar_AUC[env_rich],opalstar_AUC[env_lean]],axis=0)

            # diff 
            diffMean = meanOpalStar - meanControl

            # plot and save
            saveCrossEnv = "results/FINALFIGS/%strials%d_sims%d/crossenv/ntrials%d/%s/noptions_%d/"\
                    %(root_me,n_trials,n_states,pltn,variant,level+2)
            os.makedirs(saveCrossEnv, exist_ok=True)

            colors = (color_beta,color_alphaa)
            what = variant
            aucplotter.graph_learn(meanOpalStar,meanControl,saveCrossEnv,what)
            aucplotter.graph_diff(diffMean,saveCrossEnv,"Learning",what,abs_diff=True)
            aucplotter.graph_diff_by_AUC(np.array(meanControl),np.array(diffMean),\
                    colors,titles,saveas,maps,\
                    saveCrossEnv,"Learning",what,abs_diff=True)
        del meanControl
        del meanOpalStar
        del diffMean
        del control_AUC
        del opalstar_AUC


if __name__ == '__main__':
    main(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
