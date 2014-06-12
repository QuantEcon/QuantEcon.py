"""
Origin: QE by John Stachurski and Thomas J. Sargent
Filename: solution_lqc_ex3.py
An infinite horizon profit maximization problem for a monopolist with
adjustment costs.
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from quantecon.lqcontrol import *

# == Model parameters == #
a0      = 5
a1      = 0.5
sigma   = 0.15
rho     = 0.9
gamma   = 1
beta    = 0.95
c       = 2
T       = 120

# == Useful constants == #
m0 = (a0 - c) / (2 * a1)
m1 = 1 / (2 * a1)

# == Formulate LQ problem == #
Q = gamma
R = [[a1, -a1, 0],
     [-a1, a1, 0],
     [0,   0,  0]]
A = [[rho, 0, m0 * (1 - rho)],
     [0,   1, 0],
     [0,   0, 1]]

B = [[0],
     [1],
     [0]]
C = [[m1 * sigma],
     [0],
     [0]]

lq = LQ(Q, R, A, B, C=C, beta=beta)

# == Simulate state / control paths == #
x0 = (m0, 2, 1)
xp, up, wp = lq.compute_sequence(x0, ts_length=150)
q_bar = xp[0, :] 
q     = xp[1, :]

# == Plot simulation results == #
fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_xlabel('Time')

# == Some fancy plotting stuff -- simplify if you prefer == #
bbox = (0., 1.01, 1., .101)
legend_args = {'bbox_to_anchor' : bbox, 'loc' : 3, 'mode' : 'expand'}
p_args = {'lw' : 2, 'alpha' : 0.6}

time = range(len(q))
ax.set_xlim(0, max(time))
ax.plot(time, q_bar, 'k-', lw=2, alpha=0.6, label=r'$\bar q_t$')
ax.plot(time, q, 'b-', lw=2, alpha=0.6, label=r'$q_t$')
ax.legend(ncol=2, **legend_args)
s = r'dynamics with $\gamma = {}$'.format(gamma)
ax.text(max(time) * 0.6, 1 * q_bar.max(), s, fontsize=14)

plt.show()
