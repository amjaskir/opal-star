# Lise VANSTEEENKISTE
# internship
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank

# Name: dynamicsb.py

import numpy as np
import random
from bogacz import Bogacz
import plot_dynamicsb


def simulatebogacz(params, n_states, n_trials, p=0.5, r_mag=1, l_mag=-1):


    n_options = 1
    alpha_a, epsilon, lbda = params
    states = []

    for _ in np.arange(n_states):

        state = Bogacz(n_trials, n_options, [p], [r_mag], [l_mag])
        for t in range(n_trials):
            state.idx = t
            state.policy_forced(0)
            state.act(alpha_a, epsilon, lbda)
        states.append(state)

    return states

