"""
An permanent income / life-cycle model with polynomial growth in income
over working life followed by a fixed retirement income.  The model is solved
by combining two LQ programming problems as described in the lecture.
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from lqcontrol import *

# == Model parameters == #
r       = 0.05
beta    = 1 / (1 + r)
T       = 60
K       = 40
c_bar   = 4
sigma   = 0.35
mu      = 4
q       = 1e4
s       = 1
m1      = 2 * mu / K
m2      = - mu / K**2

# == Formulate LQ problem 1 (retirement) == #
Q = 1
R = np.zeros((4, 4)) 
Rf = np.zeros((4, 4))
Rf[0, 0] = q
A = [[1 + r, s - c_bar, 0, 0], 
     [0,     1,      0,  0],
     [0,     1,      1,  0],
     [0,     1,      2,  1]]
B = [[-1],
     [0],
     [0],
     [0]]
C = [[0],
     [0],
     [0],
     [0]]

# == Initialize LQ instance for retired agent == #
lq_retired = LQ(Q, R, A, B, C, beta=beta, T=T-K, Rf=Rf)
# == Iterate back to start of retirement, record final value function == #
for i in range(T-K):
    lq_retired.update_values()
Rf2 = lq_retired.P

# == Formulate LQ problem 2 (working life) == #
R = np.zeros((4, 4)) 
A = [[1 + r, -c_bar, m1, m2], 
     [0,     1,      0,  0],
     [0,     1,      1,  0],
     [0,     1,      2,  1]]
B = [[-1],
     [0],
     [0],
     [0]]
C = [[sigma],
     [0],
     [0],
     [0]]

# == Set up working life LQ instance with terminal Rf from lq_retired == #
lq_working = LQ(Q, R, A, B, C, beta=beta, T=K, Rf=Rf2)

# == Simulate working state / control paths == #
x0 = (0, 1, 0, 0)
xp_w, up_w, wp_w = lq_working.compute_sequence(x0)
# == Simulate retirement paths (note the initial condition) == #
xp_r, up_r, wp_r = lq_retired.compute_sequence(xp_w[:, K]) 

# == Convert results back to assets, consumption and income == #
xp = np.column_stack((xp_w, xp_r[:, 1:]))
assets = xp[0, :]               # Assets

up = np.column_stack((up_w, up_r))
c = up.flatten() + c_bar    # Consumption

time = np.arange(1, K+1)
income_w = wp_w[0, 1:K+1] + m1 * time + m2 * time**2  # Income
income_r = np.ones(T-K) * s
income = np.concatenate((income_w, income_r))

# == Plot results == #
n_rows = 2
fig, axes = plt.subplots(n_rows, 1, figsize=(12, 10))

plt.subplots_adjust(hspace=0.5)
for i in range(n_rows):
    axes[i].grid()
    axes[i].set_xlabel(r'Time')
bbox = (0., 1.02, 1., .102)
legend_args = {'bbox_to_anchor' : bbox, 'loc' : 3, 'mode' : 'expand'}
p_args = {'lw' : 2, 'alpha' : 0.7}

axes[0].plot(range(1, T+1), income, 'g-', label="non-financial income", **p_args)
axes[0].plot(range(T), c, 'k-', label="consumption", **p_args)
axes[1].plot(range(T+1), np.zeros(T+1), 'k-')
axes[0].legend(ncol=2, **legend_args)

axes[1].plot(range(T+1), assets, 'b-', label="assets", **p_args)
axes[1].plot(range(T), np.zeros(T), 'k-')
axes[1].legend(ncol=1, **legend_args)

plt.show()
