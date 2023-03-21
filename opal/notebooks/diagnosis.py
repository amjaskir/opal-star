# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank

# name: diagnosis.py
# description: takes dictionary of states over various probabilities
# and plots the G/N or Act curves at the desired trial

import numpy as np
import random
import scipy.stats as stats

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

# my dependencies
from opal import OpAL

def act_calc(rhorange,beta,G,N):
    """
    For given G and N value, calculate act values for various 
    rhorange and fixed beta 
    """

    # calculate act for various brange values 
    thresh = np.zeros(len(rhorange))
    cmp_g = np.vstack((thresh,beta*(1+rhorange)))  
    cmp_n = np.vstack((thresh,beta*(1-rhorange)))
    beta_g = np.max(cmp_g,axis=0)
    beta_n = np.max(cmp_n,axis=0)

    acts = beta_g*G - beta_n*N
    return acts

def act_curves(ameans,asem,prange,rhorange,ax):
    """
    ameans - array, rows different p value
    """
    # colors = ["red","yellow","green","blue","purple"]
    n_curves = len(rhorange) + 1
    for idx,r in enumerate(rhorange):
        c = (idx+1)/n_curves
        c = (c,c,c)
        ax.errorbar(prange,ameans[:,idx],color=c,label="rho=%.1f"%r,capsize=5)
    # ax.legend(fontsize='x-small',loc='upper left')

def gna(sims_dict,prange,trial,action,ax,noptions=1,plot_act=False,beta=1,axs=None):
    """
    sims - dict of sims["env"] = list(OpAL)
    prange - array of probability ranges to plot
    trial - trial to plot
    actions - action to plot (e.g. 0)
    """

    mean_g = [] #track probability means
    sem_g = []
    mean_n = [] 
    sem_n = []
    # rhorangeMacro = np.arange(-10,10.1)
    # rhorangeMacro = np.sort(np.hstack([np.arange(-5,5.1,1.0),np.arange(-1,1.1,.25)]))
    rhorangeMicro = np.arange(-1,1.1,.25)

    # acts = np.zeros([len(prange),len(brange)])
    print(trial)
    for pidx, p in enumerate(prange):
        env = "%d_10_%d" %(p*100,noptions)
        print(p)
        sims = sims_dict[env]
        
        collect_g = []
        collect_n = []
        for sidx, sim in enumerate(sims):
            G = sim.QG[trial,action]
            N = sim.QN[trial,action]
            collect_g.append(G)
            collect_n.append(N)

            # calculate act here for 
            if plot_act:
                # a_res_Mac = act_calc(rhorangeMacro,beta,G,N)
                a_res_Mic = act_calc(rhorangeMicro,beta,G,N)
                if sidx == 0:
                    # collect_a_Mac = a_res_Mac
                    collect_a_Mic = a_res_Mic
                else:
                    # collect_a_Mac = np.vstack((collect_a_Mac,a_res_Mac))
                    collect_a_Mic = np.vstack((collect_a_Mic,a_res_Mic))

        # for this env
        mean_g.append(np.mean(collect_g))
        sem_g.append(np.std(collect_g))
        mean_n.append(np.mean(collect_n))
        sem_n.append(np.std(collect_n))
        
        if plot_act:
            # actMac_res = np.mean(collect_a_Mac,axis=0)
            actMic_res = np.mean(collect_a_Mic,axis=0)
            # actMac_SEM = stats.sem(collect_a_Mac,axis=0)
            actMic_STD = np.std(collect_a_Mic,axis=0)
            
            if pidx == 0:
                # mean_actMac = actMac_res
                mean_actMic = actMic_res
                # sem_actMac = actMac_SEM
                sem_actMic = actMic_STD
            else:
                # mean_actMac = np.vstack((mean_actMac,actMac_res)) # each row is different p
                mean_actMic = np.vstack((mean_actMic,actMic_res))
                sem_actMic = np.vstack((sem_actMic,actMic_STD))


    linewidth = 2.
    ax.errorbar(prange,mean_g,sem_g, c = "green", linewidth = linewidth)
    ax.errorbar(prange,mean_n,sem_n, c = "red", linewidth = linewidth)
    ax.set_ylabel("G(p) N(p)")
    ax.set_xlabel("p(R)")
    ax.legend(["G(p)", "N(p)"])

    if plot_act:
        mapme = {0: "green", 1: "blue"}
        # act_curves(mean_actMac,prange,rhorangeMacro,axs[0])
        act_curves(mean_actMic,sem_actMic,prange,rhorangeMicro,axs[1])

