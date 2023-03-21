# Lise VANSTEEENKISTE
# internship
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
#
# Based off plot_dynamics.py, Alana Jaskir
# Edits by Alana Jaskir

# Name: plot_dynamicsb.py

import numpy as np
import random as rand
import matplotlib.pyplot as plt
import scipy.stats as stats


def avg_sm(states, n_trials, n_states, C, axs, color):
    """ plot average softmax probability of selecting the
    option, C, for each state

    Inputs:
    - (list of) OpAL state(s)
    - n_trials
    - n_states
    - C, plotting average softmax of choice C
    - axs, figure axis to plot on
    - color, RGB or string for line color"""
    first = True
    for state in states:
        if first:
            sms = state.SM[:, C]
            first = False
        else:
            sms = np.vstack([sms, state.SM[:, C]])

    # average over states for trial-by-trial performance
    # plot
    s = 0.01
    linewidth = 2.
    xaxis = range(n_trials)
    if n_states > 1:
        avg_sm = np.mean(sms, axis=0)
        sem_sm = stats.sem(sms, axis=0)
        axs.errorbar(xaxis, avg_sm, yerr=sem_sm, c=color, linewidth=linewidth)
    else:
        axs.errorbar(xaxis, sms, c=color, linewidth=linewidth)


def avg_qs(states, n_trials, n_states, C, axs, color_mag, style="-", flip_n=False):
    """ Plot average G and N evolution as well as estimation
    of value, V = 1/2(G - N)

    Inputs:
    - (list of) OpAL state(s)
    - n_trials
    - n_states
    - C, plotting weigth evolution of choice C
    - axs, list of figure axis to plot on
    - color, [0,1] value for color intensity
    - 
    """

    first = True
    for i in range(n_states):
        state = states[i]
        if first:
            Gs = state.QG[:, C]
            Ns = state.QN[:, C]
            Vs = state.D_g*state.QG[0:(n_trials), C] - \
            state.D_n*state.QN[0:(n_trials), C]
            first = False
        else:
            Gs = np.vstack([Gs, state.QG[:, C]])
            Ns = np.vstack([Ns, state.QN[:, C]])
            thisV = state.D_g*state.QG[0:(n_trials), C] - \
            state.D_n*state.QN[0:(n_trials), C]
            Vs = np.vstack([Vs, thisV])

    xaxis = np.arange(0, n_trials + 1)
    xaxisV = np.arange(0, n_trials)
    if n_states > 1:
        mean_gs = np.mean(Gs, axis=0)
        mean_ns = np.mean(Ns, axis=0)
        mean_vs = np.mean(Vs, axis=0)
        sem_gs = stats.sem(Gs, axis=0)
        sem_ns = stats.sem(Ns, axis=0)
        sem_vs = stats.sem(Vs, axis=0)

        if flip_n:
            mean_ns = -1*mean_ns

        axs[0].errorbar(xaxis, mean_gs, yerr=sem_gs, c=(0.0, color_mag, 0.0))
        axs[1].errorbar(xaxis, mean_ns, yerr=sem_ns, c=(color_mag, 0.0, 0.0))
        axs[2].errorbar(xaxisV, mean_vs, yerr=sem_vs, c=(color_mag, color_mag, color_mag))
    else:
        axs[0].errorbar(xaxis, Gs, c=(0.0, color_mag, 0.0))
        axs[1].errorbar(xaxis, Ns, c=(color_mag, 0.0, 0.0))
        axs[2].errorbar(xaxisV, Vs, c=(color_mag, color_mag, color_mag))

def avg_diff_qs(states, n_trials, n_states, axs, color_mag):
    """ Plot average difference in G and N evolution 
    of two options. 

    Inputs:
    - (list of) OpAL state(s)
    - n_trials
    - n_states
    - C, plotting weigth evolution of choice C
    - axs, list of figure axis to plot on
    - color, [0,1] value for color intensity
    """

    first = True
    for i in range(n_states):
        state = states[i]
        if first:
            Gs = state.QG[:, 0] - state.QG[:, 1]
            Ns = state.QN[:, 0] - state.QN[:, 1]
            first = False
        else:
            Gs = np.vstack([Gs, state.QG[:, 0] - state.QG[:, 1]])
            Ns = np.vstack([Ns, state.QN[:, 0] - state.QN[:, 1]])

    xaxis = np.arange(0, n_trials + 1)
    xaxisV = np.arange(0, n_trials)
    if n_states > 1:
        mean_gs = np.mean(Gs, axis=0)
        mean_ns = np.mean(Ns, axis=0)
        sem_gs = stats.sem(Gs, axis=0)
        sem_ns = stats.sem(Ns, axis=0)

        axs[0].errorbar(xaxis, mean_gs, yerr=sem_gs, c=(0.0, color_mag, 0.0))
        axs[1].errorbar(xaxis, mean_ns, yerr=sem_ns, c=(color_mag, 0.0, 0.0))
    else:
        axs[0].errorbar(xaxis, Gs, c=(0.0, color_mag, 0.0))
        axs[1].errorbar(xaxis, Ns, c=(color_mag, 0.0, 0.0))


def avg_vs(states, n_trials, n_states, C, axs, color_mag):

    first = True
    for state in states:
        if first:
            Vs = state.V[:, C]
            first = False
        else:
            Vs = np.vstack([Vs, state.V[:, C]])


    xaxis = np.arange(0, n_trials + 1)
    if n_states > 1:
        mean_vs = np.mean(Vs, axis=0)
        sem_vs = stats.sem(Vs, axis=0)
        axs.errorbar(xaxis, mean_vs, yerr=sem_vs, c=(color_mag, color_mag, color_mag))
    else:
        axs.errorbar(xaxis, Vs, color=(color_mag, color_mag, color_mag), s=10)


def avg_rho(states, n_trials, n_states, axs):

    first = True
    for state in states:
        if first:
            rhos = state.rho
            first = False
        else:
            rhos = np.vstack([rhos, state.rho])

    # plot
    s = 0.01
    linewidth = 2.
    xaxis = range(n_trials)
    if n_states > 1:
        # average over states for trial-by-trial performance
        avg_r = np.mean(rhos, axis=0)
        print(avg_r)
        sem_r = stats.sem(rhos, axis=0)
        axs.errorbar(xaxis, avg_r, yerr=sem_r, c="purple")
    else:
        axs.errorbar(xaxis, rhos, c="purple", linewidth=linewidth)


def avg_PE(states, n_trials, n_states, axs):

    first = True
    for state in states:
        if first:
            PEs = state.PE
            first = False
        else:
            PEs = np.vstack([PEs, state.PE])

    # plot
    s = 0.01
    linewidth = 2.
    xaxis = range(n_trials)
    if n_states > 1:
        # average over states for trial-by-trial performance
        avg_PE = np.mean(PEs, axis=0)
        sem_PE = stats.sem(PEs, axis=0)
        axs.errorbar(xaxis, avg_PE, yerr=sem_PE, c="purple")
    else:
        axs.errorbar(xaxis, PEs, c="purple", linewidth=linewidth)


def avg_ch(states,n_trials,n_states,axs,color):
    """ Plot average choice acc over trials for given
    states

    Inputs:
    - n_trials
    - n_states
    - axs, list
    - color, string
    """

    first = True
    for state in states:
        if first:
            # assumes first option is optimal option 
            # and only 2 choices
            acc = 1-state.C
            first = False
        else:
            acc = np.vstack([acc, 1-state.C])

    # plt average over states for trial-by-trial performance
    s = 0.01
    linewidth = 2.
    xaxis = range(n_trials)
    if n_states > 1:
        avg_ch = np.mean(acc, axis=0)
        sem_ch= stats.sem(acc, axis=0)
        axs.errorbar(xaxis, avg_ch, yerr=sem_ch,c=color,linewidth=linewidth)
    else:
        axs.errorbar(xaxis, acc, c=color, linewidth=linewidth)


def avg(states):
    """ Get average acc and sem across trials and across
    all states
    """
    acc = []
    for state in states:
        # assumes first option is optimal option 
        # and only 2 choices
        acc.append(np.mean(1-state.C))

    avg_acc = np.mean(acc)
    sem_acc = stats.sem(acc)
    return avg_acc, sem_acc

def avg(states):
    """ Get average acc and sem across trials and across
    all states
    """
    acc = []
    for state in states:
        # assumes first option is optimal option 
        # and only 2 choices
        acc.append(np.mean(1-state.C))

    avg_acc = np.mean(acc)
    sem_acc = stats.sem(acc)
    return avg_acc, sem_acc

def avg_da (states, n_states, k, axs, color):
    """ not quite sure what this function is doing
    """
    first = True
    for state in states:
        if first:
            choices = 1-state.C
            first = False
        else:
            choices = np.vstack([choices, 1-state.C])

    s = 0.01
    linewidth = 2.
    xaxis=np.array([k]*n_states)
    if n_states > 1:
        avg_da = np.mean(choices, axis=1)
        sem_da = stats.sem(choices, axis=1)
        axs.errorbar(xaxis, avg_da, yerr=sem_da, c=color, linewidth=linewidth)
    else:
        axs.errorbar(xaxis, choices, c=color, linewidth=linewidth)
