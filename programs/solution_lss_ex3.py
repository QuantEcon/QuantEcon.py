
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from lss import LSS
import random

phi_1, phi_2, phi_3, phi_4 = 0.5, -0.2, 0, 0.5
sigma = 0.1

A = [[phi_1, phi_2, phi_3, phi_4],
     [1,     0,     0,     0],
     [0,     1,     0,     0],
     [0,     0,     1,     0]]
C = [sigma, 0, 0, 0]
G = [1, 0, 0, 0]

I = 20
T = 50
ar = LSS(A, C, G, mu_0=np.ones(4))
ymin, ymax = -0.5, 1.15

fig, ax = plt.subplots()

ax.set_ylim(ymin, ymax)
ax.set_xlabel(r'time', fontsize=16)
ax.set_ylabel(r'$y_t$', fontsize=16)

ensemble_mean = np.zeros(T)
for i in range(I):
    x, y = ar.simulate(ts_length=T)
    y = y.flatten()
    ax.plot(y, 'c-', lw=0.8, alpha=0.5)
    ensemble_mean = ensemble_mean + y

ensemble_mean = ensemble_mean / I
ax.plot(ensemble_mean, color='b', lw=2, alpha=0.8, label=r'$\bar y_t$')

m = ar.moment_sequence()
population_means = []
for t in range(T):
    mu_x, mu_y, Sigma_x, Sigma_y = m.next()
    population_means.append(mu_y)
ax.plot(population_means, color='g', lw=2, alpha=0.8, label=r'$G\mu_t$')
ax.legend(ncol=2)

plt.show()
