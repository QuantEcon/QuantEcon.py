"""
An LQ permanent income / life-cycle model with hump-shaped income

    y_t = m1 * t + m2 * t^2 + sigma w_{t+1}

where {w_t} is iid N(0, 1) and the coefficients m1 and m2 are chosen so that
p(t) = m1 * t + m2 * t^2 has an inverted U shape with p(0) = 0, p(T/2) = mu
and p(T) = 0.
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from lqcontrol import *

# == Model parameters == #
r       = 0.05
beta    = 1 / (1 + r)
T       = 50
c_bar   = 1.5
sigma   = 0.15
mu      = 2
q       = 1e4
m1      = T * (mu / (T/2)**2)
m2      = - (mu / (T/2)**2)

# == Formulate as an LQ problem == #
Q = 1
R = np.zeros((4, 4)) 
Rf = np.zeros((4, 4))
Rf[0, 0] = q
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

# == Compute solutions and simulate == #
lq = LQ(Q, R, A, B, C, beta=beta, T=T, Rf=Rf)
x0 = (0, 1, 0, 0)
xp, up, wp = lq.compute_sequence(x0)

# == Convert results back to assets, consumption and income == #
ap = xp[0, :]               # Assets
c = up.flatten() + c_bar    # Consumption
time = np.arange(1, T+1)
income = wp[0, 1:] + m1 * time + m2 * time**2  # Income


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

axes[1].plot(range(T+1), ap.flatten(), 'b-', label="assets", **p_args)
axes[1].plot(range(T), np.zeros(T), 'k-')
axes[1].legend(ncol=1, **legend_args)

plt.show()


