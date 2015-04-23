"""
Plots consumption, income and debt for the simple infinite horizon LQ
permanent income model with Gaussian iid income.
"""


import random
import numpy as np
import matplotlib.pyplot as plt

r       = 0.05
beta    = 1 / (1 + r)
T       = 60
sigma   = 0.15
mu = 1


def time_path():
    w = np.random.randn(T+1)  # w_0, w_1, ..., w_T
    w[0] = 0
    b = np.zeros(T+1)
    for t in range(1, T+1):
        b[t] = w[1:t].sum()
    b = - sigma * b
    c = mu + (1 - beta) * (sigma * w - b)
    return w, b, c


# == Figure showing a typical realization == #

if 1:
    fig, ax = plt.subplots()

    p_args = {'lw': 2, 'alpha': 0.7}
    ax.grid()
    ax.set_xlabel(r'Time')
    bbox = (0., 1.02, 1., .102)
    legend_args = {'bbox_to_anchor': bbox, 'loc': 'upper left',
                   'mode': 'expand'}

    w, b, c = time_path()
    ax.plot(list(range(T+1)), mu + sigma * w, 'g-',
            label="non-financial income", **p_args)
    ax.plot(list(range(T+1)), c, 'k-', label="consumption", **p_args)
    ax.plot(list(range(T+1)), b, 'b-', label="debt", **p_args)
    ax.legend(ncol=3, **legend_args)

    plt.show()

# == Figure showing multiple consumption paths == #

if 0:
    fig, ax = plt.subplots()

    p_args = {'lw': 0.8, 'alpha': 0.7}
    ax.grid()
    ax.set_xlabel(r'Time')
    ax.set_ylabel(r'Consumption')
    b_sum = np.zeros(T+1)
    for i in range(250):
        rcolor = random.choice(('c', 'g', 'b', 'k'))
        w, b, c = time_path()
        ax.plot(list(range(T+1)), c, color=rcolor, **p_args)

    plt.show()
