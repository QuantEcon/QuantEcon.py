
import numpy as np
import matplotlib.pyplot as plt
from quantecon import LinearStateSpace
import random

phi_1, phi_2, phi_3, phi_4 = 0.5, -0.2, 0, 0.5
sigma = 0.1

A = [[phi_1, phi_2, phi_3, phi_4],
     [1,     0,     0,     0],
     [0,     1,     0,     0],
     [0,     0,     1,     0]]
C = [sigma, 0, 0, 0]
G = [1, 0, 0, 0]

T = 30
ar = LinearStateSpace(A, C, G, mu_0=np.ones(4))

ymin, ymax = -0.8, 1.25

fig, axes = plt.subplots(1, 2, figsize=(8, 3))

for ax in axes:
    ax.grid(alpha=0.4)

ax = axes[0]

ax.set_ylim(ymin, ymax)
ax.set_ylabel(r'$y_t$', fontsize=16)
ax.vlines((T,), -1.5, 1.5)

ax.set_xticks((T,))
ax.set_xticklabels((r'$T$',))

sample = []
for i in range(20):
    rcolor = random.choice(('c', 'g', 'b', 'k'))
    x, y = ar.simulate(ts_length=T+15)
    y = y.flatten()
    ax.plot(y, color=rcolor, lw=1, alpha=0.5)
    ax.plot((T,), (y[T],), 'ko', alpha=0.5)
    sample.append(y[T])

y = y.flatten()
axes[1].set_ylim(ymin, ymax)
axes[1].hist(sample, bins=16, normed=True, orientation='horizontal', alpha=0.5)

plt.show()
