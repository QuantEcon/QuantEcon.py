"""
Filename: lq_permanent_1.py
Authors: John Stachurski and Thomas J. Sargent

A permanent income / life-cycle model with iid income
"""

import numpy as np
import matplotlib.pyplot as plt
from quantecon import LQ

# == Model parameters == #
r       = 0.05
beta    = 1 / (1 + r)
T       = 45
c_bar   = 2
sigma   = 0.25
mu      = 1
q       = 1e6

# == Formulate as an LQ problem == #
Q = 1
R = np.zeros((2, 2))
Rf = np.zeros((2, 2))
Rf[0, 0] = q
A = [[1 + r, -c_bar + mu],
     [0,     1]]
B = [[-1],
     [0]]
C = [[sigma],
     [0]]

# == Compute solutions and simulate == #
lq = LQ(Q, R, A, B, C, beta=beta, T=T, Rf=Rf)
x0 = (0, 1)
xp, up, wp = lq.compute_sequence(x0)

# == Convert back to assets, consumption and income == #
assets = xp[0, :]           # a_t
c = up.flatten() + c_bar    # c_t
income = wp[0, 1:] + mu     # y_t

# == Plot results == #
n_rows = 2
fig, axes = plt.subplots(n_rows, 1, figsize=(12, 10))

plt.subplots_adjust(hspace=0.5)
for i in range(n_rows):
    axes[i].grid()
    axes[i].set_xlabel(r'Time')
bbox = (0., 1.02, 1., .102)
legend_args = {'bbox_to_anchor': bbox, 'loc': 3, 'mode': 'expand'}
p_args = {'lw': 2, 'alpha': 0.7}

axes[0].plot(list(range(1, T+1)), income, 'g-', label="non-financial income",
             **p_args)
axes[0].plot(list(range(T)), c, 'k-', label="consumption", **p_args)
axes[0].legend(ncol=2, **legend_args)

axes[1].plot(list(range(1, T+1)), np.cumsum(income - mu), 'r-',
             label="cumulative unanticipated income", **p_args)
axes[1].plot(list(range(T+1)), assets, 'b-', label="assets", **p_args)
axes[1].plot(list(range(T)), np.zeros(T), 'k-')
axes[1].legend(ncol=2, **legend_args)

plt.show()
